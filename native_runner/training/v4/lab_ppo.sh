#!/usr/bin/env bash
# One machine, no Slurm: engines, worker server, topology and trainer in the background, cleaned up on exit.
#
#   lab_ppo.sh start NAME [UPDATES]   start a fresh run in runs/NAME (logs in runs/NAME/logs; the trainer
#                                     writes metrics and checkpoints to a new runs/NAME/train-* per start)
#   lab_ppo.sh resume NAME [UPDATES]  continue runs/NAME from its newest strict checkpoint
#   lab_ppo.sh status [NAME]          processes, latest metrics, checkpoints and free disk
#   lab_ppo.sh load [NAME]            live CPU (per core group) and GPU use, and where updates spend time
#   lab_ppo.sh stop [NAME MINUTES]    stop the run and clean up engines, workers and /dev/shm; with NAME and
#                                     MINUTES, first wait up to MINUTES for the next strict checkpoint
#   lab_ppo.sh schedule NAME HH:MM HH:MM [RETRY_MINUTES]   every day, resume NAME at the first time and
#                                     stop it by the second (waiting for a checkpoint from 12 minutes before);
#                                     for RETRY_MINUTES after the start (default: until the stop begins),
#                                     check every 15 minutes and resume again if the run died without being
#                                     stopped; replaces any earlier schedule
#   lab_ppo.sh ensure NAME            resume NAME unless it is running or was stopped on purpose
#   lab_ppo.sh unschedule             remove the daily schedule
#
# UPDATES is the total update count to stop at (default 30000, more than a week). The trainer options
# default to the sparring configuration in docs/lab_training_zh.md and CR_PPO_EXTRA_ARGS replaces them;
# a resume reuses the options saved in runs/NAME/trainer-options.txt. Other settings:
# CR_PPO_ENGINE_COUNT (8), CR_LAB_ENGINE_CPUS (0-7), CR_LAB_WORKER_CPUS (8-15), CR_LAB_CONTROL_CPUS (16-19), CR_LAB_TRAIN_GPUS
# (0; nvidia-smi numbers, e.g. 0,1), CR_LAB_KEEP_CHECKPOINTS (3) and CR_LAB_CHECKPOINT_BUCKET (1000):
# the newest few strict checkpoints are kept plus the first one in every bucket of that many updates.
# NCCL_P2P_DISABLE, CR_PPO_INFERENCE_BATCH_WAIT_MS and other environment variables pass through to the trainer.
# A start saves these launch settings to runs/NAME/launch-env and a resume restores them, unless the same
# variable is given on the resume command line.
set -euo pipefail

launch_vars=(CR_PPO_ENGINE_COUNT CR_LAB_ENGINE_CPUS CR_LAB_WORKER_CPUS CR_LAB_CONTROL_CPUS CR_LAB_TRAIN_GPUS
  CR_PPO_INFERENCE_BATCH_WAIT_MS CR_PPO_SEGMENT_SECONDS CR_PPO_LANE_MICROBATCH_SIZE NCCL_P2P_DISABLE
  CR_PPO_CHECKPOINT)
explicit_vars=" "
for variable in "${launch_vars[@]}" CR_PPO_EXTRA_ARGS; do
  [[ ! -v $variable ]] || explicit_vars+="$variable "
done

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "$script_dir/../../.." && pwd)
# Relative paths (CR_PPO_CHECKPOINT in launch-env, trainer options) mean the repository root, also under cron.
cd "$repository_root"
source "${repository_root}/native_runner/local_config.sh"
python=${CR_AI_PYTHON:-python}
runs=$repository_root/runs
node_dir=$runs/lab-node
engine_cpus=${CR_LAB_ENGINE_CPUS:-0-7}
worker_cpus=${CR_LAB_WORKER_CPUS:-8-15}
control_cpus=${CR_LAB_CONTROL_CPUS:-16-19}
engine_count=${CR_PPO_ENGINE_COUNT:-8}
base_port=${CR_PPO_BASE_PORT:-28000}
worker_port=${CR_PPO_REMOTE_WORKER_PORT:-29900}
train_gpus=${CR_LAB_TRAIN_GPUS:-0}
keep_checkpoints=${CR_LAB_KEEP_CHECKPOINTS:-3}
checkpoint_bucket=${CR_LAB_CHECKPOINT_BUCKET:-1000}
host=$(hostname -s)
user=$(id -un)
export SLURM_JOB_ID=local TMPDIR=${TMPDIR:-/dev/shm}
export CR_PPO_ENGINES_PER_GUEST=${CR_PPO_ENGINES_PER_GUEST:-4}

default_extra_args() {
  # The path to Active IL contains a space, and the trainer options are split on whitespace.
  ln -sfn "$repository_root/checkpoints/active IL/checkpoint-step-00000030.pt" "$runs/active-il.pt"
  echo "--execution-delay-ticks 18-24 --min-play-gap-ticks 4-8 --self-play-percent 40 --fixed-opponent-percent 60" \
    "--fixed-opponent-checkpoint checkpoints/General/checkpoint-step-00000460.pt" \
    "--fixed-opponent-checkpoint $runs/active-il.pt" \
    "--fixed-opponent-checkpoint checkpoints/2_6hog_expert/hog26-specialist1.pt" \
    "--il-anchor-checkpoint checkpoints/IL/checkpoint-step-00029396.pt" \
    "--personal-elixir-overflow-penalty 0.01 --personal-elixir-overflow-grace 1" \
    "--gate-temperature-start 0.5 --gate-temperature-end 0.2 --gate-temperature-anneal-updates 3000" \
    "--snapshot-every 200"
}

cpu_count() {
  local part count=0
  for part in ${1//,/ }; do
    count=$((count + ${part#*-} - ${part%-*} + 1))
  done
  echo "$count"
}

supervisor_pids() {
  pgrep -u "$user" -f "lab_ppo\.sh _run " || true
}

log() {
  echo "$(date -Is) $*"
}

# Only this user's processes, and only emulators of the PPO engine clusters (AVD names crppo_*).
cleanup() {
  [[ ! -d $node_dir ]] || touch "$node_dir/STOP_ENGINES"
  pkill -u "$user" -f "train_ppo_self_play_cluster|torch\.distributed\.run" 2>/dev/null || true
  for _ in {1..6}; do
    pgrep -u "$user" -f "launch_ppo_engine_node" >/dev/null || break
    sleep 5
  done
  local pattern="launch_ppo_engine_node|remote_ppo_worker_server|tcp_endpoint_proxy|train_ppo_self_play_cluster"
  pkill -u "$user" -f "$pattern" 2>/dev/null || true
  pkill -u "$user" -f -- "-avd crppo_" 2>/dev/null || true
  sleep 3
  pkill -9 -u "$user" -f "$pattern" 2>/dev/null || true
  pkill -9 -u "$user" -f -- "-avd crppo_" 2>/dev/null || true
  rm -rf /dev/shm/firstlight-ppo-local-*
}

prune_checkpoints() {
  local directory=$1 files index update
  mapfile -t files < <(find "$directory" -maxdepth 1 -name 'update-*.pt' 2>/dev/null | sort)
  local -A kept_buckets=()
  for ((index = 0; index < ${#files[@]} - keep_checkpoints; index++)); do
    update=$(basename "${files[index]}" .pt)
    update=$((10#${update#update-} / checkpoint_bucket))
    if [[ -z ${kept_buckets[$update]:-} ]]; then
      kept_buckets[$update]=1
    else
      rm -f "${files[index]}"
    fi
  done
}

# The trainer only writes into an empty directory, so every start or resume gets its own train-* stage.
latest_stage() {
  { find "$1" -mindepth 1 -maxdepth 1 -type d -name 'train-*' 2>/dev/null || true; } | sort | tail -n 1
}

newest_checkpoint() {
  { find "$1" -path '*/train-*/checkpoints/update-*.pt' 2>/dev/null || true; } | sort | tail -n 1
}

wait_for() {
  local description=$1 seconds=$2 pid=$3
  shift 3
  local deadline=$((SECONDS + seconds))
  until "$@"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      log "FAILED: $description exited before it was ready"
      return 1
    fi
    if ((SECONDS > deadline)); then
      log "FAILED: $description was not ready after ${seconds}s"
      return 1
    fi
    sleep 5
  done
}

run() {
  local run_dir=$1 updates=$2 resume=$3
  local logs=$run_dir/logs stage
  stage=$run_dir/train-$(date +%Y%m%d-%H%M%S)
  trap cleanup EXIT
  trap 'log "stop requested"; exit 130' INT TERM
  if [[ -z $resume && ! -f ${CR_PPO_CHECKPOINT:-} ]]; then
    log "FAILED: the starting checkpoint CR_PPO_CHECKPOINT='${CR_PPO_CHECKPOINT:-}' is not a file"
    exit 1
  fi
  log "starting from ${resume:-${CR_PPO_CHECKPOINT}}"
  log "cleaning up leftovers"
  cleanup
  rm -rf "$node_dir"
  mkdir -p "$logs"

  log "starting $engine_count engines on CPUs $engine_cpus"
  # More engines than engine CPUs: the emulators time-share them (each still gets one vCPU per engine).
  (($(cpu_count "$engine_cpus") >= engine_count)) || export CR_PPO_ENGINE_CPU_SHARING=1
  taskset -c "$engine_cpus" bash "$script_dir/launch_ppo_engine_node.sh" "$node_dir" 0 "$engine_count" "$base_port" \
    >"$logs/engines.log" 2>&1 &
  local engine_pid=$!
  wait_for "the engine node" "${CR_LAB_ENGINE_TIMEOUT:-1800}" "$engine_pid" \
    grep -q ENGINE_CLUSTER_READY "$logs/engines.log"

  log "starting the worker server on CPUs $worker_cpus"
  local sharing=()
  # More engines than worker CPUs: let the per-engine workers time-share the worker cores.
  (($(cpu_count "$worker_cpus") >= engine_count)) || sharing=(--allow-worker-cpu-sharing)
  taskset -c "$worker_cpus" "$python" -m native_runner.training.v4.remote_ppo_worker_server \
    --bind-host "$host" --port "$worker_port" --global-start 0 --engine-count "$engine_count" \
    --engine-base-port "$base_port" "${sharing[@]}" >"$logs/worker.log" 2>&1 &
  local worker_pid=$!
  wait_for "the worker server" 120 "$worker_pid" \
    bash -c "ss -ltnH 'sport = :$worker_port' | grep -q LISTEN"

  local gpu_count
  gpu_count=$(tr ',' '\n' <<<"$train_gpus" | grep -c .)
  "$python" -m native_runner.training.v4.build_ppo_multinode_topology \
    --node-ready "$node_dir/engine-node-ready.json" --output "$run_dir/topology.json" \
    --trainer-job-id local --trainer-host "$host" --gpus "$gpu_count" \
    --worker-cpus "$worker_cpus" --control-cpus "$control_cpus" >"$logs/topology.log" 2>&1

  local extra=${CR_PPO_EXTRA_ARGS:-}
  if [[ -z $extra && -n $resume && -s $run_dir/trainer-options.txt ]]; then
    # A resume must repeat the options the run started with, even if the defaults changed since.
    extra=$(<"$run_dir/trainer-options.txt")
  fi
  extra=${extra:-$(default_extra_args)}
  echo "$extra" >"$run_dir/trainer-options.txt"
  log "training to update $updates on GPU $train_gpus into $stage${resume:+ from $resume}"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$train_gpus CR_PPO_EXTRA_ARGS=$extra \
    bash "$script_dir/launch_ppo_multinode_training.sh" "$stage" "$run_dir/topology.json" "$updates" \
    "${CR_PPO_SEGMENT_SECONDS:-40}" "$resume" >>"$logs/trainer.log" 2>&1 &
  local trainer_pid=$!

  while kill -0 "$trainer_pid" 2>/dev/null; do
    sleep 60 &
    wait $! || true
    prune_checkpoints "$stage/checkpoints"
    if ! kill -0 "$engine_pid" 2>/dev/null || ! kill -0 "$worker_pid" 2>/dev/null; then
      log "FAILED: the engine node or the worker server exited; see engines.log and worker.log"
      save_guest_logs "$logs"
      exit 1
    fi
    if (($(df -Pk "$run_dir" | awk 'NR == 2 {print $4}') < 5 * 1024 * 1024)); then
      log "WARNING: less than 5 GB of disk left"
    fi
  done
  local status=0
  wait "$trainer_pid" || status=$?
  prune_checkpoints "$stage/checkpoints"
  log "trainer exited with status $status; newest checkpoint: $(newest_checkpoint "$run_dir")"
  ((status == 0)) || save_guest_logs "$logs"
  exit "$status"
}

# The probe reports why it rejected a capture only in the guests' logcat, which is gone once the emulators
# stop, so keep the tail of each guest's log after a failure.
save_guest_logs() {
  local adb=${CR_TRAINING_ADB:-adb} serial directory
  directory=$1/guest-logcat-$(date +%Y%m%d-%H%M%S)
  mkdir -p "$directory"
  for serial in $("$adb" devices 2>/dev/null | awk '$1 ~ /^emulator-/ && $2 == "device" {print $1}'); do
    timeout 30 "$adb" -s "$serial" logcat -d -t 5000 >"$directory/$serial.txt" 2>&1 || true
  done
  log "saved guest logcat to $directory"
}

save_launch_env() {
  local variable
  for variable in "${launch_vars[@]}"; do
    [[ ! -v $variable ]] || echo "$variable=${!variable}"
  done >"$1/launch-env"
}

load_launch_env() {
  local key value
  [[ -f $1/launch-env ]] || return 0
  while IFS='=' read -r key value; do
    [[ $explicit_vars == *" $key "* ]] || export "$key=$value"
  done <"$1/launch-env"
}

# Wait up to $2 minutes for a strict checkpoint newer than the current one in runs/$1.
wait_for_checkpoint() {
  local before deadline=$((SECONDS + $2 * 60))
  before=$(newest_checkpoint "$runs/$1")
  while ((SECONDS < deadline)) && [[ -n $(supervisor_pids) ]]; do
    [[ $(newest_checkpoint "$runs/$1") == "$before" ]] || { log "new checkpoint $(newest_checkpoint "$runs/$1")"; return; }
    sleep 15
  done
  log "no new checkpoint within $2 minutes; stopping anyway"
}

# Present after a deliberate stop (by hand or by the schedule) until the next start or resume, so
# ensure can tell a run that was stopped from one that crashed.
stop_marker=$runs/lab-stopped-on-purpose

start() {
  local name=${1:?usage: lab_ppo.sh start|resume NAME [UPDATES]} updates=${2:-30000} resume=${3:-}
  if [[ -n $(supervisor_pids) ]]; then
    echo "a lab PPO run is already active; run: lab_ppo.sh stop" >&2
    exit 1
  fi
  rm -f "$stop_marker"
  local run_dir=$runs/$name
  mkdir -p "$run_dir/logs"
  save_launch_env "$run_dir"
  setsid nohup bash "$script_dir/lab_ppo.sh" _run "$run_dir" "$updates" "$resume" \
    >>"$run_dir/logs/supervisor.log" 2>&1 </dev/null &
  echo "started run $name in the background (supervisor pid $!)"
  echo "  progress:  tail -f $run_dir/logs/supervisor.log $run_dir/logs/trainer.log"
  echo "  status:    bash native_runner/training/v4/lab_ppo.sh status $name"
  echo "  stop:      bash native_runner/training/v4/lab_ppo.sh stop"
}

case ${1:-} in
  start)
    shift
    if [[ -n $(latest_stage "$runs/${1:?usage: lab_ppo.sh start NAME [UPDATES]}") ]]; then
      echo "runs/$1 already holds a run; pick a new NAME or use: lab_ppo.sh resume $1" >&2
      exit 1
    fi
    start "$1" "${2:-30000}"
    ;;
  resume)
    shift
    checkpoint=$(newest_checkpoint "$runs/${1:?usage: lab_ppo.sh resume NAME [UPDATES]}")
    load_launch_env "$runs/$1"
    if [[ -n $checkpoint ]]; then
      echo "resuming from $checkpoint"
      start "$1" "${2:-30000}" "$checkpoint"
    elif [[ -s $runs/$1/trainer-options.txt && -f ${CR_PPO_CHECKPOINT:-} ]]; then
      # The run never reached a strict checkpoint: start it again from its own starting point and options.
      echo "runs/$1 has no strict checkpoint yet; starting it again from $CR_PPO_CHECKPOINT"
      [[ $explicit_vars == *" CR_PPO_EXTRA_ARGS "* ]] || export CR_PPO_EXTRA_ARGS="$(<"$runs/$1/trainer-options.txt")"
      start "$1" "${2:-30000}"
    else
      echo "runs/$1 has no strict checkpoint to resume from" >&2
      exit 1
    fi
    ;;
  stop)
    mkdir -p "$runs" && date -Is >"$stop_marker"
    if [[ -n ${3:-} ]]; then
      log "waiting up to $3 minutes for the next checkpoint of $2"
      wait_for_checkpoint "$2" "$3"
    fi
    pids=$(supervisor_pids)
    if [[ -n $pids ]]; then
      kill -TERM $pids
      for _ in {1..30}; do
        [[ -n $(supervisor_pids) ]] || break
        sleep 5
      done
    fi
    cleanup
    echo "stopped; remaining processes: $(pgrep -u "$user" -fc -- "-avd crppo_|remote_ppo_worker_server|train_ppo_self_play_cluster" || true)"
    ;;
  schedule)
    name=${2:?usage: lab_ppo.sh schedule NAME HH:MM HH:MM} begin=${3:?start time HH:MM} end=${4:?stop time HH:MM}
    [[ $begin =~ ^([01]?[0-9]|2[0-3]):([0-5][0-9])$ ]] || { echo "bad start time $begin" >&2; exit 2; }
    begin_minute=$((10#${BASH_REMATCH[1]} * 60 + 10#${BASH_REMATCH[2]}))
    [[ $end =~ ^([01]?[0-9]|2[0-3]):([0-5][0-9])$ ]] || { echo "bad stop time $end" >&2; exit 2; }
    end_minute=$((10#${BASH_REMATCH[1]} * 60 + 10#${BASH_REMATCH[2]}))
    drain_minute=$(((end_minute - 12 + 1440) % 1440))
    command -v crontab >/dev/null || { echo "crontab is not installed" >&2; exit 1; }
    script=$script_dir/lab_ppo.sh
    # By default keep checking all day, until the checkpoint-aware stop begins.
    retry_minutes=${5:-$(((drain_minute - begin_minute + 1440) % 1440 - 15))}
    log_file=$runs/schedule.log
    {
      { crontab -l 2>/dev/null || true; } | grep -v "# lab_ppo schedule" || true
      echo "$((begin_minute % 60)) $((begin_minute / 60)) * * * bash $script resume $name >>$log_file 2>&1 # lab_ppo schedule"
      echo "$((drain_minute % 60)) $((drain_minute / 60)) * * * bash $script stop $name 11 >>$log_file 2>&1 # lab_ppo schedule"
      echo "$((end_minute % 60)) $((end_minute / 60)) * * * bash $script stop >>$log_file 2>&1 # lab_ppo schedule"
      for ((offset = 15; offset <= retry_minutes; offset += 15)); do
        retry_minute=$(((begin_minute + offset) % 1440))
        echo "$((retry_minute % 60)) $((retry_minute / 60)) * * * bash $script ensure $name >>$log_file 2>&1 # lab_ppo schedule"
      done
    } | crontab -
    echo "scheduled $name daily: resume at $begin, retry for $retry_minutes minutes, stop by $end (log: $log_file)"
    crontab -l | grep "# lab_ppo schedule"
    ;;
  ensure)
    name=${2:?usage: lab_ppo.sh ensure NAME}
    if [[ -n $(supervisor_pids) ]]; then
      log "ensure $name: running"
    elif [[ -e $stop_marker ]]; then
      log "ensure $name: stopped on purpose at $(<"$stop_marker"); not restarting"
    else
      log "ensure $name: not running and not stopped on purpose; resuming"
      exec bash "$script_dir/lab_ppo.sh" resume "$name"
    fi
    ;;
  unschedule)
    { crontab -l 2>/dev/null || true; } | { grep -v "# lab_ppo schedule" || true; } | crontab -
    echo "daily schedule removed"
    ;;
  status)
    echo "supervisor: $(supervisor_pids | tr '\n' ' ')"
    echo "emulators: $(pgrep -u "$user" -fc -- "qemu-system.*-avd crppo_" || true)  free disk: $(df -Ph "$runs" | awk 'NR == 2 {print $4}')"
    if [[ -n ${2:-} ]]; then
      run_dir=$runs/$2
      stage=$(latest_stage "$run_dir")
      tail -n 3 "$run_dir/logs/supervisor.log" 2>/dev/null || true
      "$python" - "$stage/metrics.jsonl" 2>/dev/null <<'PY' || echo "no metrics yet"
import json, sys
lines = open(sys.argv[1]).read().splitlines()[-5:]
for line in lines:
    m = json.loads(line)
    print(f"update {m['update']}: {m['end_to_end_seconds']:.1f}s  frames/s {m['end_to_end_owner_frames_per_second']:.0f}  "
          f"entropy {m.get('entropy', float('nan')):.3f}  rejected {m['native_rejected_actions']}")
PY
      "$python" - "$run_dir" 2>/dev/null <<'PY' || echo "no finished games yet"
import json, statistics, sys
from pathlib import Path
stages = sorted(Path(sys.argv[1]).glob("train-*"))
rows = [json.loads(line) for stage in stages if (stage / "matches.jsonl").exists()
        for line in (stage / "matches.jsonl").read_text().splitlines()]
names = {}
for stage in stages:
    if (stage / "league.json").exists():
        for entry in json.loads((stage / "league.json").read_text())["entries"]:
            names[entry["checkpoint_id"]] = Path(entry["checkpoint_path"]).stem
recent = rows[-256:]
seats = [(r, o) for r in recent for o in r["current_owners"]]
first = sorted(r["first_nonwait_action_tick"][o] / 20 for r, o in seats if r["first_nonwait_action_tick"][o] is not None)
q = lambda p: first[min(len(first) - 1, int(p * len(first)))]
print(f"{len(rows)} finished games; last {len(recent)}: learner first play 10%/median/90% = "
      f"{q(.1):.0f}/{q(.5):.0f}/{q(.9):.0f}s, plays per game {statistics.mean(r['active_actions'][o] for r, o in seats):.1f}")
# Whole-game elixir wasted at full, comparable to the replays' elixir_leaked (expert winners: median ~2.3).
leaked = sorted(r["cumulative_elixir_overflow_wasted"][o] for r, o in seats if r.get("cumulative_elixir_overflow_wasted"))
if leaked:
    k = lambda p: leaked[min(len(leaked) - 1, int(p * len(leaked)))]
    print(f"  learner elixir wasted per game 10%/median/90% = {k(.1):.1f}/{k(.5):.1f}/{k(.9):.1f}")
# Only tracked when --game-elixir-overflow-penalty is on: waste after the learner's first card.
after = sorted(r["game_elixir_overflow_wasted"][o] for r, o in seats if any(r.get("game_elixir_overflow_wasted") or ()))
if after:
    a = lambda p: after[min(len(after) - 1, int(p * len(after)))]
    print(f"  ... after the first card 10%/median/90% = {a(.1):.1f}/{a(.5):.1f}/{a(.9):.1f}")
# Win rate against each frozen opponent, earliest games versus latest; self-play is always about 50%.
fixed = [r for r in rows if r["policy_matchup"] == "current-history" and r.get("opponent_checkpoint_id")]
for opponent in sorted({r["opponent_checkpoint_id"] for r in fixed}, key=lambda value: names.get(value, value)):
    games = [r["result"][r["current_owners"][0]] for r in fixed if r["opponent_checkpoint_id"] == opponent]
    # Draws (result 0) are neither wins nor losses; show them so they don't read as losses.
    rate = lambda part: (
        f"{100 * sum(value > 0 for value in part) / len(part):.0f}% win "
        f"{100 * sum(value == 0 for value in part) / len(part):.0f}% draw of {len(part)}"
    ) if part else "-"
    window = min(300, len(games) // 2)
    print(f"  vs {names.get(opponent, opponent[:12]):32s} win rate: first {rate(games[:window])}  latest {rate(games[-window:] if window else games)}")
PY
      echo "checkpoints: $(find "$run_dir" -path '*/train-*/checkpoints/update-*.pt' 2>/dev/null | wc -l), newest $(newest_checkpoint "$run_dir")"
    fi
    ;;
  load)
    # Live CPU use per core group and GPU use over 5 seconds, then where recent updates spent their time.
    "$python" - "$engine_cpus" "$worker_cpus" "$control_cpus" <<'PY'
import sys, time

def parse(spec):
    cores = []
    for part in spec.split(","):
        start, _, end = part.partition("-")
        cores.extend(range(int(start), int(end or start) + 1))
    return cores

def sample():
    rows = {}
    for line in open("/proc/stat"):
        name, *values = line.split()
        if name.startswith("cpu") and name != "cpu":
            values = [int(value) for value in values]
            rows[int(name[3:])] = (sum(values), values[3] + values[4])
    return rows

before = sample()
time.sleep(5)
after = sample()
busy = {core: 100 * (1 - (after[core][1] - before[core][1]) / max(after[core][0] - before[core][0], 1)) for core in after}
named = {}
for label, spec in zip(("engines", "workers", "trainer"), sys.argv[1:]):
    cores = parse(spec)
    named.update(dict.fromkeys(cores, label))
    values = [busy[core] for core in cores if core in busy]
    if not values:
        continue
    print(f"CPU {label:8s} {spec:>6s}: mean {sum(values) / len(values):3.0f}%  per core " + " ".join(f"{value:.0f}" for value in values))
other = [busy[core] for core in busy if core not in named]
if other:
    print(f"CPU other    {'':>6s}: mean {sum(other) / len(other):3.0f}%")
PY
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null \
      | sed 's/^/GPU /' || echo "nvidia-smi not available"
    if [[ -n ${2:-} ]]; then
      "$python" - "$(latest_stage "$runs/$2")/metrics.jsonl" 2>/dev/null <<'PY' || echo "no metrics yet"
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]).read().splitlines()[-10:]]
mean = lambda key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows)
total, collect, optimize = mean("end_to_end_seconds"), mean("collection_seconds"), mean("optimizer_seconds")
model, assembly = mean("collection_model_seconds"), mean("collection_assembly_seconds")
waiting = max(collect - model - assembly - mean("collection_action_seconds") - mean("collection_command_seconds"), 0.0)
share = lambda value: f"{value:5.1f}s {100 * value / total:3.0f}%"
print(f"last {len(rows)} updates, mean {total:.1f}s per update:")
print(f"  collection            {share(collect)}")
print(f"    GPU inference       {share(model)}   ({mean('collection_inference_batches'):.0f} batches of ~{mean('collection_inference_mean_actor_rows'):.0f} rows)")
print(f"    trainer batching    {share(assembly)}   (CPU, trainer cores)")
print(f"    waiting for engines {share(waiting)}   (CPU, engine and worker cores)")
print(f"  optimization (GPU)    {share(optimize)}")
print(f"  per engine: native step {mean('collection_native_step_seconds'):.1f}s, observation parsing {mean('collection_tensorize_seconds'):.1f}s")
PY
    fi
    ;;
  _run)
    shift
    run "$@"
    ;;
  *)
    sed -n '2,/^set /{/^#/p}' "$0"
    exit 2
    ;;
esac
