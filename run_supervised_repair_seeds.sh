#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
if (( $# > 0 )); then
  OUTPUT_ROOT="$1"
  shift
else
  OUTPUT_ROOT="${SCRIPT_DIR}/supervised/retraining_seeds"
fi
if (( $# > 0 )); then
  SEEDS=("$@")
else
  SEEDS=(0 1 2 3 4)
fi

mkdir -p "${OUTPUT_ROOT}"
if [[ -n "${GPU_IDS:-}" ]]; then
  read -r -a GPUS <<< "${GPU_IDS}"
  if (( ${#GPUS[@]} < ${#SEEDS[@]} )); then
    echo "GPU_IDS must contain at least one GPU index per seed" >&2
    exit 2
  fi
else
  GPUS=()
fi

PIDS=()
for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$index]}"
  output_dir="${OUTPUT_ROOT}/seed${seed}"
  mkdir -p "${output_dir}"
  echo "Starting complete CEGIS run for retraining seed ${seed}: ${output_dir}"
  if (( ${#GPUS[@]} )); then
    CUDA_VISIBLE_DEVICES="${GPUS[$index]}" \
      "${SCRIPT_DIR}/run_supervised_repair_loop.sh" "${seed}" "${output_dir}" \
      > "${output_dir}/cegis.log" 2>&1 &
  else
    "${SCRIPT_DIR}/run_supervised_repair_loop.sh" "${seed}" "${output_dir}" \
      > "${output_dir}/cegis.log" 2>&1 &
  fi
  PIDS+=("$!")
  echo "$!" > "${output_dir}/worker.pid"
done

failed=0
for index in "${!SEEDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    echo "Seed ${SEEDS[$index]} completed successfully"
  else
    status=$?
    echo "Seed ${SEEDS[$index]} failed with status ${status}" >&2
    failed=1
  fi
done

if (( failed )); then
  exit 1
fi
echo "Completed retraining seeds: ${SEEDS[*]}"
