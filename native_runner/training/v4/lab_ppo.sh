#!/usr/bin/env bash
# One machine, no Slurm: engines, worker server, topology and trainer in the background, cleaned up on exit.
#
#   lab_ppo.sh start NAME [UPDATES]   start a fresh run in runs/NAME (logs in runs/NAME/logs; the trainer
#                                     writes metrics and checkpoints to a new runs/NAME/train-* per start)
#   lab_ppo.sh resume NAME [UPDATES]  continue runs/NAME from its newest strict checkpoint; if its trainer options
#                                     changed since that checkpoint (see options), warm-start a new stage from it
#   lab_ppo.sh options NAME [--FLAG [VALUE...]]... [--unset --FLAG]...   show runs/NAME's trainer options, or
#                                     set, replace (every occurrence of a repeatable flag) or remove some;
#                                     the next resume, by hand or by the schedule, uses them
#   lab_ppo.sh status [NAME]          processes, latest metrics, checkpoints and free disk
#   lab_ppo.sh load [NAME]            live CPU (per core group) and GPU use, and where updates spend time
#   lab_ppo.sh trend NAME [BLOCK]     per stage and every BLOCK (16) updates: win rate against the fixed
#                                     opponents, PPO step statistics and play habits, and the options changed
#   lab_ppo.sh stop [NAME MINUTES]    stop the run and clean up engines, workers and /dev/shm; with NAME and
#                                     MINUTES, first wait up to MINUTES for the next strict checkpoint
#   lab_ppo.sh schedule NAME HH:MM HH:MM [RETRY_MINUTES]   every day, resume NAME at the first time and
#                                     stop it by the second (waiting for a checkpoint from 12 minutes before);
#                                     for RETRY_MINUTES after the start (default: until the stop begins),
#                                     check every 15 minutes and resume again if the run died without being
#                                     stopped; replaces any earlier schedule
#   lab_ppo.sh ensure NAME            resume NAME unless it is running or was stopped on purpose
#   lab_ppo.sh watch NAME [MINUTES]   around the clock, run ensure NAME every MINUTES (default 5), so a run
#                                     that dies without being stopped comes back on its own
#   lab_ppo.sh unwatch NAME           remove that check
#   lab_ppo.sh unschedule             remove the daily schedule
#
# UPDATES is the total update count to stop at (default 30000, more than a week). The trainer options
# default to the sparring configuration in docs/lab_training_zh.md and CR_PPO_EXTRA_ARGS replaces them;
# a resume reuses the options saved in runs/NAME/trainer-options.txt (CR_PPO_EXTRA_ARGS on a resume, or the
# options command, replaces them; the changed options then take effect through a warm start: the newest
# strict checkpoint's weights and optimizer state, a fresh update count). Other settings:
# CR_PPO_ENGINE_COUNT (8), CR_LAB_ENGINE_CPUS (0-7), CR_LAB_WORKER_CPUS (8-15), CR_LAB_CONTROL_CPUS (16-19), CR_LAB_TRAIN_GPUS
# (0; nvidia-smi numbers, e.g. 0,1), CR_LAB_KEEP_CHECKPOINTS (3) and CR_LAB_CHECKPOINT_BUCKET (1000):
# the newest few strict checkpoints are kept plus the first one in every bucket of that many updates.
# NCCL_P2P_DISABLE, CR_PPO_INFERENCE_BATCH_WAIT_MS and other environment variables pass through to the trainer.
# CR_LAB_INSTANCE (unset, or 1-3) runs a second, independent lab on the same machine: its own engine and
# worker ports, emulator names and console ports, /dev/shm files, node directory, stop marker and schedule,
# and every command (stop, status, ensure, schedule) acts on that instance only. Give each instance its own
# CPU ranges and GPU (CR_LAB_*_CPUS, CR_LAB_TRAIN_GPUS); a run remembers its instance in launch-env.
# The trainer runs on CR_LAB_CONTROL_CPUS only (CR_PPO_TRAINER_CPUS=control); control-and-engines also lets
# it use the worker CPUs, where busy workers preempt its serial inference thread.
# A start saves these launch settings to runs/NAME/launch-env and a resume restores them, unless the same
# variable is given on the resume command line.
set -euo pipefail

launch_vars=(CR_PPO_ENGINE_COUNT CR_LAB_ENGINE_CPUS CR_LAB_WORKER_CPUS CR_LAB_CONTROL_CPUS CR_LAB_TRAIN_GPUS
  CR_PPO_INFERENCE_BATCH_WAIT_MS CR_PPO_SEGMENT_SECONDS CR_PPO_LANE_MICROBATCH_SIZE NCCL_P2P_DISABLE
  CR_PPO_CHECKPOINT CR_LAB_INSTANCE CR_PPO_TRAINER_CPUS)
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
engine_cpus=${CR_LAB_ENGINE_CPUS:-0-7}
worker_cpus=${CR_LAB_WORKER_CPUS:-8-15}
control_cpus=${CR_LAB_CONTROL_CPUS:-16-19}
engine_count=${CR_PPO_ENGINE_COUNT:-8}
train_gpus=${CR_LAB_TRAIN_GPUS:-0}
keep_checkpoints=${CR_LAB_KEEP_CHECKPOINTS:-3}
checkpoint_bucket=${CR_LAB_CHECKPOINT_BUCKET:-1000}
host=$(hostname -s)
user=$(id -un)
export TMPDIR=${TMPDIR:-/dev/shm}
export CR_PPO_ENGINES_PER_GUEST=${CR_PPO_ENGINES_PER_GUEST:-4}
export CR_PPO_TRAINER_CPUS=${CR_PPO_TRAINER_CPUS:-control}

# Ports given by the caller, read once: instance_setup exports its own choice, and a later call (after
# launch-env selects another instance) must not mistake that export for the caller's.
requested_base_port=${CR_PPO_BASE_PORT:-}
requested_worker_port=${CR_PPO_REMOTE_WORKER_PORT:-}

# Everything two labs on one machine must not share. The default instance keeps the original names.
instance_setup() {
  instance=${CR_LAB_INSTANCE:-}
  [[ -z $instance || $instance =~ ^[1-3]$ ]] || { echo "CR_LAB_INSTANCE must be unset or 1-3" >&2; exit 2; }
  suffix=${instance:+-$instance}
  node_dir=$runs/lab-node$suffix
  stop_marker=$runs/lab-stopped-on-purpose$suffix
  schedule_tag="# lab_ppo schedule${instance:+ instance $instance}"
  base_port=${requested_base_port:-$((28000 + 200 * ${instance:-0}))}
  worker_port=${requested_worker_port:-$((29900 + ${instance:-0}))}
  guest_prefix=crppo$instance
  export SLURM_JOB_ID=local$instance CR_PPO_GUEST_PREFIX=$guest_prefix
  export CR_EMULATOR_BASE_PORT=$((5554 + 16 * ${instance:-0})) CR_PPO_REMOTE_WORKER_PORT=$worker_port
}
instance_setup

# Extended-regex form of a literal path or value.
regex_literal() {
  sed 's/[][\\.*^$+?(){}|/]/\\&/g' <<<"$1"
}

# PIDs of this user's processes whose command line matches $1 and, if given, not $2.
matching_pids() {
  local include=$1 exclude=${2:-}
  { pgrep -u "$user" -af -- "$include" 2>/dev/null || true; } \
    | { if [[ -n $exclude ]]; then grep -Ev -- "$exclude" || true; else cat; fi; } \
    | awk '$1 != '"$$"' {print $1}'
}

kill_matching() {
  local signal=$1 pids
  shift
  pids=$(matching_pids "$@")
  [[ -z $pids ]] || kill "-$signal" $pids 2>/dev/null || true
}

# This instance's processes, by the ports, paths and names only it uses. The default instance also matches
# trainers started before topologies moved into the node directory (runs/NAME/topology.json).
instance_patterns() {
  local node runs_re
  node=$(regex_literal "$node_dir")
  runs_re=$(regex_literal "$runs")
  echo "launch_ppo_engine_node\.sh $node( |$)"
  echo "remote_ppo_worker_server .*--port $worker_port( |$)"
  echo "tcp_endpoint_proxy.* --listen-base $base_port( |$)"
  echo "-avd ${guest_prefix}_"
  if [[ -n $instance ]]; then
    echo "(train_ppo_self_play_cluster|torch\.distributed\.run).*--engine-ready $node/topology\.json"
  else
    echo "(train_ppo_self_play_cluster|torch\.distributed\.run).*--engine-ready $runs_re/[^ ]*topology\.json"
  fi
}

default_extra_args() {
  # The path to Active IL contains a space, and the trainer options are split on whitespace.
  ln -sfn "$repository_root/checkpoints/active IL/checkpoint-step-00000030.pt" "$runs/active-il.pt"
  echo "--execution-delay-ticks 18-24 --min-play-gap-ticks 6-10 --self-play-percent 40 --fixed-opponent-percent 60" \
    "--fixed-opponent-checkpoint checkpoints/General/checkpoint-step-00000460.pt" \
    "--fixed-opponent-checkpoint $runs/active-il.pt" \
    "--fixed-opponent-checkpoint checkpoints/2_6hog_expert/hog26-specialist1.pt" \
    "--il-anchor-checkpoint checkpoints/IL/checkpoint-step-00029396.pt" \
    "--personal-elixir-overflow-penalty 0 --personal-elixir-overflow-grace 0" \
    "--game-elixir-overflow-penalty 0.01 --game-elixir-overflow-grace 3" \
    "--early-first-play-penalty 0 --early-first-play-free-seconds 6" \
    "--fixed-opponent-opening-force-after-seconds 6-20 --spell-king-activation-penalty 0.1" \
    "--fixed-opponent-opening-natural-percent 20" \
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
  if [[ -n $instance ]]; then
    matching_pids "lab_ppo\.sh _run .* lab-instance=$instance( |$)"
  else
    matching_pids "lab_ppo\.sh _run " "lab-instance=[1-9]"
  fi
}

log() {
  echo "$(date -Is) $*"
}

# Only this user's processes, and only emulators of the PPO engine clusters (AVD names crppo_*).
# Only this instance's processes: see instance_patterns.
cleanup() {
  local -a patterns
  mapfile -t patterns < <(instance_patterns)
  local trainer=${patterns[4]} engine=${patterns[0]} exclude=""
  [[ -n $instance ]] || exclude="lab-node-[1-9]/topology"
  [[ ! -d $node_dir ]] || touch "$node_dir/STOP_ENGINES"
  kill_matching TERM "$trainer" "$exclude"
  for _ in {1..6}; do
    [[ -n $(matching_pids "$engine") ]] || break
    sleep 5
  done
  local signal pattern
  for signal in TERM KILL; do
    for pattern in "${patterns[@]}"; do
      kill_matching "$signal" "$pattern" "$exclude"
    done
    [[ $signal == KILL ]] || sleep 3
  done
  rm -rf /dev/shm/firstlight-ppo-"$SLURM_JOB_ID"-*
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

# The options a stage trained with; stages from before this record were the current options.
stage_options() {
  local file=$1/options-$(basename "$2").txt
  if [[ -f $file ]]; then cat "$file"; else cat "$1/trainer-options.txt"; fi
}

# Keep what the newest stage trained with before its options change, so the next resume can tell.
remember_stage_options() {
  local stage
  stage=$(newest_checkpoint "$1")
  [[ -n $stage ]] || return 0
  stage=$(dirname "$(dirname "$stage")")
  [[ -f $1/options-$(basename "$stage").txt ]] || cp "$1/trainer-options.txt" "$1/options-$(basename "$stage").txt"
}

edit_options() {
  local run_dir=$1 joined
  shift
  [[ -s $run_dir/trainer-options.txt ]] || { echo "$run_dir has no trainer options; start the run first" >&2; exit 1; }
  local -a current new=() drop=() kept=()
  read -ra current <"$run_dir/trainer-options.txt"
  while (($#)); do
    if [[ $1 == --unset ]]; then
      [[ ${2:-} == --* ]] || { echo "usage: --unset --FLAG" >&2; exit 2; }
      drop+=("$2")
      shift 2
    elif [[ $1 == --* ]]; then
      drop+=("$1")
      new+=("$1")
      shift
      while (($#)) && [[ $1 != --* ]]; do new+=("$1"); shift; done
    else
      echo "expected --FLAG, got '$1'" >&2
      exit 2
    fi
  done
  local token skipping=0
  for token in "${current[@]}"; do
    if [[ $token == --* ]]; then
      skipping=0
      [[ " ${drop[*]} " != *" $token "* ]] || skipping=1
    fi
    ((skipping)) || kept+=("$token")
  done
  remember_stage_options "$run_dir"
  joined="${kept[*]} ${new[*]}"
  joined=${joined# }
  echo "${joined% }" >"$run_dir/trainer-options.txt"
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
    --node-ready "$node_dir/engine-node-ready.json" --output "$node_dir/topology.json" \
    --trainer-job-id "$SLURM_JOB_ID" --trainer-host "$host" --gpus "$gpu_count" \
    --worker-cpus "$worker_cpus" --control-cpus "$control_cpus" >"$logs/topology.log" 2>&1
  # The trainer reads the node's copy, whose path identifies this instance's trainer to cleanup.
  cp "$node_dir/topology.json" "$run_dir/topology.json"

  local extra=${CR_PPO_EXTRA_ARGS:-}
  if [[ -z $extra && -n $resume && -s $run_dir/trainer-options.txt ]]; then
    # A resume must repeat the options the run started with, even if the defaults changed since.
    extra=$(<"$run_dir/trainer-options.txt")
  fi
  extra=${extra:-$(default_extra_args)}
  echo "$extra" >"$run_dir/trainer-options.txt"
  echo "$extra" >"$run_dir/options-$(basename "$stage").txt"
  log "training to update $updates on GPU $train_gpus into $stage${resume:+ from $resume}"
  # A warm start from this run's own checkpoint keeps its optimizer state (not saved as a run option).
  local trainer_extra=$extra
  [[ -z ${CR_LAB_WARM_START_OPTIMIZER:-} ]] || trainer_extra+=" --warm-start-optimizer"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$train_gpus CR_PPO_EXTRA_ARGS=$trainer_extra \
    bash "$script_dir/launch_ppo_multinode_training.sh" "$stage" "$node_dir/topology.json" "$updates" \
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

# Act on the instance a run was started in, unless CR_LAB_INSTANCE is given on the command line.
use_run_instance() {
  local saved
  [[ $explicit_vars != *" CR_LAB_INSTANCE "* && -f $runs/$1/launch-env ]] || return 0
  saved=$(sed -n 's/^CR_LAB_INSTANCE=//p' "$runs/$1/launch-env")
  export CR_LAB_INSTANCE=$saved
  instance_setup
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

# A deliberate stop (by hand or by the schedule) leaves $stop_marker until the next start or resume, so
# ensure can tell a run that was stopped from one that crashed.

start() {
  local name=${1:?usage: lab_ppo.sh start|resume NAME [UPDATES]} updates=${2:-30000} resume=${3:-}
  if [[ -n $(supervisor_pids) ]]; then
    echo "a lab PPO run is already active${instance:+ in instance $instance}; run:" \
      "${instance:+CR_LAB_INSTANCE=$instance }lab_ppo.sh stop" >&2
    exit 1
  fi
  rm -f "$stop_marker"
  local run_dir=$runs/$name
  mkdir -p "$run_dir/logs"
  save_launch_env "$run_dir"
  setsid nohup bash "$script_dir/lab_ppo.sh" _run "$run_dir" "$updates" "$resume" "lab-instance=${instance:-0}" \
    >>"$run_dir/logs/supervisor.log" 2>&1 </dev/null &
  echo "started run $name in the background (supervisor pid $!)"
  echo "  progress:  tail -f $run_dir/logs/supervisor.log $run_dir/logs/trainer.log"
  echo "  status:    bash native_runner/training/v4/lab_ppo.sh status $name"
  echo "  stop:      ${instance:+CR_LAB_INSTANCE=$instance }bash native_runner/training/v4/lab_ppo.sh stop"
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
    instance_setup
    if [[ $explicit_vars == *" CR_PPO_EXTRA_ARGS "* && -s $runs/$1/trainer-options.txt ]]; then
      remember_stage_options "$runs/$1"
      echo "$CR_PPO_EXTRA_ARGS" >"$runs/$1/trainer-options.txt"
    fi
    if [[ -n $checkpoint ]] && [[ $(stage_options "$runs/$1" "$(dirname "$(dirname "$checkpoint")")") \
      != "$(cat "$runs/$1/trainer-options.txt" 2>/dev/null)" ]]; then
      echo "trainer options changed since $checkpoint; warm-starting a new stage from it with:"
      cat "$runs/$1/trainer-options.txt"
      export CR_PPO_CHECKPOINT=$checkpoint CR_PPO_EXTRA_ARGS="$(<"$runs/$1/trainer-options.txt")"
      export CR_LAB_WARM_START_OPTIMIZER=1
      start "$1" "${2:-30000}"
    elif [[ -n $checkpoint ]]; then
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
    [[ -z ${2:-} ]] || { use_run_instance "$2"; }
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
    echo "stopped${instance:+ instance $instance}; remaining processes: $(instance_patterns | while read -r pattern; do matching_pids "$pattern"; done | sort -u | wc -l)"
    ;;
  schedule)
    name=${2:?usage: lab_ppo.sh schedule NAME HH:MM HH:MM} begin=${3:?start time HH:MM} end=${4:?stop time HH:MM}
    [[ $begin =~ ^([01]?[0-9]|2[0-3]):([0-5][0-9])$ ]] || { echo "bad start time $begin" >&2; exit 2; }
    begin_minute=$((10#${BASH_REMATCH[1]} * 60 + 10#${BASH_REMATCH[2]}))
    [[ $end =~ ^([01]?[0-9]|2[0-3]):([0-5][0-9])$ ]] || { echo "bad stop time $end" >&2; exit 2; }
    end_minute=$((10#${BASH_REMATCH[1]} * 60 + 10#${BASH_REMATCH[2]}))
    drain_minute=$(((end_minute - 12 + 1440) % 1440))
    command -v crontab >/dev/null || { echo "crontab is not installed" >&2; exit 1; }
    if [[ -z $(newest_checkpoint "$runs/$name") && ! -s $runs/$name/trainer-options.txt ]]; then
      echo "runs/$name has no checkpoint or saved options, so the daily resume would fail; runs with checkpoints:" >&2
      { find "$runs" -path '*/train-*/checkpoints/update-*.pt' 2>/dev/null || true; } \
        | sed "s|^$runs/||; s|/.*||" | sort -u | sed 's/^/  /' >&2
      exit 1
    fi
    use_run_instance "$name"
    script="${instance:+CR_LAB_INSTANCE=$instance }bash $script_dir/lab_ppo.sh"
    # By default keep checking all day, until the checkpoint-aware stop begins.
    retry_minutes=${5:-$(((drain_minute - begin_minute + 1440) % 1440 - 15))}
    log_file=$runs/schedule$suffix.log
    {
      { crontab -l 2>/dev/null || true; } | { grep -v -- "$schedule_tag\$" || true; }
      echo "$((begin_minute % 60)) $((begin_minute / 60)) * * * $script resume $name >>$log_file 2>&1 $schedule_tag"
      echo "$((drain_minute % 60)) $((drain_minute / 60)) * * * $script stop $name 11 >>$log_file 2>&1 $schedule_tag"
      echo "$((end_minute % 60)) $((end_minute / 60)) * * * $script stop $name 0 >>$log_file 2>&1 $schedule_tag"
      for ((offset = 15; offset <= retry_minutes; offset += 15)); do
        retry_minute=$(((begin_minute + offset) % 1440))
        echo "$((retry_minute % 60)) $((retry_minute / 60)) * * * $script ensure $name >>$log_file 2>&1 $schedule_tag"
      done
    } | crontab -
    echo "scheduled $name daily: resume at $begin, retry for $retry_minutes minutes, stop by $end (log: $log_file)"
    crontab -l | grep -- "$schedule_tag\$"
    ;;
  options)
    name=${2:?usage: lab_ppo.sh options NAME [--FLAG [VALUE...]]... [--unset --FLAG]...}
    use_run_instance "$name"
    if (($# > 2)); then
      edit_options "$runs/$name" "${@:3}"
      echo "saved; the next resume of $name warm-starts from its newest checkpoint with these options"
      [[ -z $(supervisor_pids) ]] || echo "(the running trainer keeps its options until it is stopped)"
    fi
    tr ' ' '\n' <"$runs/$name/trainer-options.txt" | awk '/^--/ {if (line) print line; line = $0; next} {line = line " " $0} END {print line}'
    ;;
  ensure)
    name=${2:?usage: lab_ppo.sh ensure NAME}
    use_run_instance "$name"
    if [[ -n $(supervisor_pids) ]]; then
      log "ensure $name: running"
    elif [[ -e $stop_marker ]]; then
      log "ensure $name: stopped on purpose at $(<"$stop_marker"); not restarting"
    else
      log "ensure $name: not running and not stopped on purpose; resuming"
      exec bash "$script_dir/lab_ppo.sh" resume "$name"
    fi
    ;;
  watch)
    name=${2:?usage: lab_ppo.sh watch NAME [MINUTES]} minutes=${3:-5}
    [[ $minutes =~ ^[1-9][0-9]?$ ]] && ((minutes <= 30)) || { echo "MINUTES must be 1-30" >&2; exit 2; }
    command -v crontab >/dev/null || { echo "crontab is not installed" >&2; exit 1; }
    if [[ -z $(newest_checkpoint "$runs/$name") && ! -s $runs/$name/trainer-options.txt ]]; then
      echo "runs/$name has no checkpoint or saved options to resume from" >&2
      exit 1
    fi
    watch_tag="# lab_ppo watch $name"
    {
      { crontab -l 2>/dev/null || true; } | { grep -Ev -- " $(regex_literal "$watch_tag")\$" || true; }
      echo "*/$minutes * * * * bash $script_dir/lab_ppo.sh ensure $name >>$runs/$name/logs/watch.log 2>&1 $watch_tag"
    } | crontab -
    echo "watching $name: every $minutes minutes, resume it if it died without being stopped (log: runs/$name/logs/watch.log)"
    echo "a deliberate stop (lab_ppo.sh stop) is left alone; remove the check with: lab_ppo.sh unwatch $name"
    ;;
  unwatch)
    name=${2:?usage: lab_ppo.sh unwatch NAME}
    { crontab -l 2>/dev/null || true; } | { grep -Ev -- " $(regex_literal "# lab_ppo watch $name")\$" || true; } | crontab -
    echo "no longer watching $name"
    ;;
  unschedule)
    { crontab -l 2>/dev/null || true; } | { grep -v -- "$schedule_tag\$" || true; } | crontab -
    echo "daily schedule removed${instance:+ for instance $instance}"
    ;;
  status)
    [[ -z ${2:-} ]] || use_run_instance "$2"
    echo "${instance:+instance $instance  }supervisor: $(supervisor_pids | tr '\n' ' ')"
    echo "emulators: $(pgrep -u "$user" -fc -- "qemu-system.*-avd ${guest_prefix}_" || true)  free disk: $(df -Ph "$runs" | awk 'NR == 2 {print $4}')"
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
# Only tracked when --early-first-play-penalty is on: games the learner opened before the free time.
early = [r["early_first_play_penalty_total"][o] for r, o in seats if "early_first_play_penalty_total" in r]
if early and any(early):
    print(f"  opened the game early: {sum(value > 0 for value in early) / len(early):.0%} of games, "
          f"mean penalty {statistics.mean(early):.3f}")
# Only tracked when --late-opening-penalty is on: games the learner leaked past the grace before its first card.
late = [r["late_opening_penalty_total"][o] for r, o in seats if "late_opening_penalty_total" in r]
if late and any(late):
    print(f"  late opening charged: {sum(value > 0 for value in late) / len(late):.0%} of games, "
          f"mean penalty {statistics.mean(late):.3f}")
# Only with --fixed-opponent-opening-force-after-seconds: fixed opponents made to open the game.
frozen = [r for r in recent if r["policy_matchup"] == "current-history"]
eligible = [r for r in frozen if "opening_force_exempt" in r
            and (r["opening_force_exempt"] or r.get("opening_force_deadline_seconds") is not None)]
if eligible:
    print(f"  fixed opponents exempt from opening timeout: "
          f"{sum(r['opening_force_exempt'] for r in eligible) / len(eligible):.0%} of their games")
forced = [r["opening_forced_tick"] / 20 for r in frozen if r.get("opening_forced_owner") is not None]
if forced:
    print(f"  fixed opponents forced to open: {len(forced) / len(frozen):.0%} of their games, "
          f"median at {statistics.median(forced):.0f}s")
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
  trend)
    # Win rate against the fixed opponents over the whole run, next to what the optimizer did at the time.
    name=${2:?usage: lab_ppo.sh trend NAME [BLOCK]}
    [[ ${3:-16} =~ ^[1-9][0-9]*$ ]] || { echo "BLOCK must be a positive number of updates (one NAME per command)" >&2; exit 2; }
    [[ -d $runs/$name ]] || { echo "no run named $name in $runs" >&2; exit 1; }
    "$python" - "$runs/$name" "${3:-16}" <<'PY'
import json, math, statistics as st, sys
from pathlib import Path
run, block = Path(sys.argv[1]), int(sys.argv[2])
previous = None
shown = 0
for stage in sorted(run.glob("train-*")):
    if not (stage / "metrics.jsonl").exists():
        continue
    metrics = [json.loads(line) for line in (stage / "metrics.jsonl").read_text().splitlines()]
    games = [json.loads(line) for line in (stage / "matches.jsonl").read_text().splitlines()] \
        if (stage / "matches.jsonl").exists() else []
    if not metrics:
        continue
    shown += 1
    print(f"== {stage.name}  updates {metrics[0]['update']}-{metrics[-1]['update']}")
    saved = run / f"options-{stage.name}.txt"
    options = set(saved.read_text().split("--")) if saved.exists() else None
    if options is not None and previous is not None and options != previous:
        print("   options -", " | ".join(sorted(item.strip() for item in previous - options if item.strip())))
        print("   options +", " | ".join(sorted(item.strip() for item in options - previous if item.strip())))
    previous = options if options is not None else previous
    print("   updates    vs fixed (n)  self  kl      clip   EV     entropy  adv scale  steps  plays  wasted  first"
          "  grad cos  snr")
    for low in range(metrics[0]["update"], metrics[-1]["update"] + 1, block):
        high = low + block - 1
        part = [row for row in metrics if low <= row["update"] <= high]
        played = [row for row in games if low <= row["update"] <= high]
        fixed = [row["result"][row["current_owners"][0]] for row in played if row.get("opponent_checkpoint_id")]
        seats = [(row, owner) for row in played for owner in row["current_owners"]]
        median = lambda key: st.median(float(row[key]) for row in part if key in row) \
            if any(key in row for row in part) else math.nan
        wasted = [row["cumulative_elixir_overflow_wasted"][owner] for row, owner in seats
                  if row.get("cumulative_elixir_overflow_wasted")]
        first = [row["first_nonwait_action_tick"][owner] / 20 for row, owner in seats
                 if row["first_nonwait_action_tick"][owner] is not None]
        rate = f"{100 * sum(value > 0 for value in fixed) / len(fixed):3.0f}% ({len(fixed):3d})" if fixed else "    -     "
        print(f"   {low:4d}-{high:<4d}  {rate}  {len(played) - len(fixed):4d}  {median('approx_kl'):.4f}  "
              f"{median('clip_fraction'):.3f}  {median('value_explained_variance'):5.2f}  {median('entropy'):.3f}    "
              f"{median('advantage_scale'):7.3f}  {median('minibatch_updates'):5.0f}  "
              f"{st.mean(row['active_actions'][owner] for row, owner in seats) if seats else math.nan:5.1f}  "
              f"{st.median(wasted) if wasted else math.nan:6.1f}  {st.median(first) if first else math.nan:5.0f}  "
              f"{median('gradient_half_cosine'):8.3f}  {median('gradient_snr'):6.2f}")
if not shown:
    stages = sorted(path.name for path in run.glob("train-*"))
    print(f"{run.name}: no finished update yet" + (f" (stages: {', '.join(stages)})" if stages else " (no train-* stage; did it start?)"))
PY
    ;;
  load)
    # Live CPU use per core group and GPU use over 5 seconds, then where recent updates spent their time.
    if [[ -n ${2:-} ]]; then
      # Label the core groups with the ranges that run was started with.
      load_launch_env "$runs/$2"
      engine_cpus=${CR_LAB_ENGINE_CPUS:-$engine_cpus}
      worker_cpus=${CR_LAB_WORKER_CPUS:-$worker_cpus}
      control_cpus=${CR_LAB_CONTROL_CPUS:-$control_cpus}
    fi
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
