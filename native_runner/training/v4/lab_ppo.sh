#!/usr/bin/env bash
# One machine, no Slurm: engines, worker server, topology and trainer in the background, cleaned up on exit.
#
#   lab_ppo.sh start NAME [UPDATES]   start a fresh run in runs/NAME (logs in runs/NAME/logs; the trainer
#                                     writes metrics and checkpoints to a new runs/NAME/train-* per start)
#   lab_ppo.sh resume NAME [UPDATES]  continue runs/NAME from its newest strict checkpoint
#   lab_ppo.sh status [NAME]          processes, latest metrics, checkpoints and free disk
#   lab_ppo.sh load [NAME]            live CPU (per core group) and GPU use, and where updates spend time
#   lab_ppo.sh stop                   stop the run and clean up engines, workers and /dev/shm
#
# UPDATES is the total update count to stop at (default 30000, more than a week). The trainer options
# default to the sparring configuration in docs/lab_training_zh.md and CR_PPO_EXTRA_ARGS replaces them;
# a resume reuses the options saved in runs/NAME/trainer-options.txt. Other settings:
# CR_LAB_ENGINE_CPUS (0-7), CR_LAB_WORKER_CPUS (8-15), CR_LAB_CONTROL_CPUS (16-19), CR_LAB_TRAIN_GPUS
# (0; nvidia-smi numbers, e.g. 0,1), CR_LAB_KEEP_CHECKPOINTS (3) and CR_LAB_CHECKPOINT_BUCKET (1000):
# the newest few strict checkpoints are kept plus the first one in every bucket of that many updates.
# NCCL_P2P_DISABLE and other environment variables pass through to the trainer.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "$script_dir/../../.." && pwd)
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
  echo "--execution-delay-ticks 18-24 --self-play-percent 40 --fixed-opponent-percent 60" \
    "--fixed-opponent-checkpoint checkpoints/General/checkpoint-step-00000460.pt" \
    "--fixed-opponent-checkpoint $runs/active-il.pt" \
    "--fixed-opponent-checkpoint checkpoints/2_6hog_expert/hog26-specialist1.pt" \
    "--il-anchor-checkpoint checkpoints/IL/checkpoint-step-00029396.pt" \
    "--personal-elixir-overflow-penalty 0.01 --personal-elixir-overflow-grace 1" \
    "--gate-temperature-start 0.5 --gate-temperature-end 0.2 --gate-temperature-anneal-updates 3000" \
    "--snapshot-every 200"
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
  log "cleaning up leftovers"
  cleanup
  rm -rf "$node_dir"
  mkdir -p "$logs"

  log "starting $engine_count engines on CPUs $engine_cpus"
  taskset -c "$engine_cpus" bash "$script_dir/launch_ppo_engine_node.sh" "$node_dir" 0 "$engine_count" "$base_port" \
    >"$logs/engines.log" 2>&1 &
  local engine_pid=$!
  wait_for "the engine node" "${CR_LAB_ENGINE_TIMEOUT:-1800}" "$engine_pid" \
    grep -q ENGINE_CLUSTER_READY "$logs/engines.log"

  log "starting the worker server on CPUs $worker_cpus"
  taskset -c "$worker_cpus" "$python" -m native_runner.training.v4.remote_ppo_worker_server \
    --bind-host "$host" --port "$worker_port" --global-start 0 --engine-count "$engine_count" \
    --engine-base-port "$base_port" >"$logs/worker.log" 2>&1 &
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
  exit "$status"
}

start() {
  local name=${1:?usage: lab_ppo.sh start|resume NAME [UPDATES]} updates=${2:-30000} resume=${3:-}
  if [[ -n $(supervisor_pids) ]]; then
    echo "a lab PPO run is already active; run: lab_ppo.sh stop" >&2
    exit 1
  fi
  local run_dir=$runs/$name
  mkdir -p "$run_dir/logs"
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
    if [[ -z $checkpoint ]]; then
      echo "runs/$1 has no strict checkpoint to resume from" >&2
      exit 1
    fi
    echo "resuming from $checkpoint"
    start "$1" "${2:-30000}" "$checkpoint"
    ;;
  stop)
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
      "$python" - "$stage/matches.jsonl" 2>/dev/null <<'PY' || echo "no finished games yet"
import json, statistics, sys
rows = [json.loads(line) for line in open(sys.argv[1]).read().splitlines()[-256:]]
seats = [(r, o) for r in rows for o in r["current_owners"]]
first = sorted(r["first_nonwait_action_tick"][o] / 20 for r, o in seats if r["first_nonwait_action_tick"][o] is not None)
if first:
    q = lambda p: first[min(len(first) - 1, int(p * len(first)))]
    print(f"last {len(rows)} games: learner first play 10%/median/90% = {q(.1):.0f}/{q(.5):.0f}/{q(.9):.0f}s, "
          f"plays per game {statistics.mean(r['active_actions'][o] for r, o in seats):.1f}")
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
