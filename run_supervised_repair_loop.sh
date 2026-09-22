#!/usr/bin/env bash
set -euo pipefail

echo "Running example: ACC SUPERVISED"
echo "BEWARE: Before running this you need to build NCubeV and have the supervised dependencies available"

start_time=$(date +%s)

print_total_runtime() {
  local end_time elapsed
  end_time=$(date +%s)
  elapsed=$((end_time - start_time))
  echo "Total runtime: ${elapsed}s"
}

trap print_total_runtime EXIT

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
RETRAIN_SEED="${1:-42}"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/supervised}"
INITIAL_MODEL="${SCRIPT_DIR}/supervised/ppo_acc_bigger_200000_steps.zip"
INITIAL_POLYTOPES="${SCRIPT_DIR}/supervised/acc_bigger_polytopes.pkl"
LOG_DIR="${OUTPUT_DIR}/logs"

# Make the loop self-contained when launched from a fresh shell.
source "${SCRIPT_DIR}/env.sh"

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
else
  echo "Error: neither 'python' nor 'python3' is available in PATH."
  exit 1
fi


mkdir -p "${LOG_DIR}"

run_ncubev () {
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./NCubeV/deps/NCubeV/bin/NCubeV NCubeV/test/parsing/examples/acc/formula NCubeV/test/parsing/examples/acc/fixed NCubeV/test/parsing/examples/acc/mapping "${OUTPUT_DIR}/${1}.onnx" "${OUTPUT_DIR}/${2}" --approx 1
}

echo "Retraining seed: ${RETRAIN_SEED}"
echo "Output directory: ${OUTPUT_DIR}"
run_nb=1
while true; do
  log_file="${LOG_DIR}/supervised_repair_run_${run_nb}.log"
  echo "Starting run ${run_nb}. Logs: ${log_file}"

  (
    cd "${SCRIPT_DIR}"
    echo "[Run ${run_nb}] Starting supervised retraining"
    PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u acc_supervised_retrain.py "${run_nb}" \
      --seed "${RETRAIN_SEED}" \
      --output-dir "${OUTPUT_DIR}" \
      --initial-model "${INITIAL_MODEL}" \
      --initial-polytopes "${INITIAL_POLYTOPES}"
  ) > "${log_file}" 2>&1

  echo "Run ${run_nb} retraining completed. Running NCubeV."
  if (
    cd "${SCRIPT_DIR}"
    echo "[Run ${run_nb}] Starting NCubeV verification"
    run_ncubev "ppo_acc_bigger_200000_steps-${run_nb}" "acc_bigger_polytopes-${run_nb}"
  ) >> "${log_file}" 2>&1; then
    echo "Run ${run_nb} formally certified by NCubeV. Stopping loop."
    break
  else
    ncube_status=$?
    if [[ "${ncube_status}" -ne 1 ]]; then
      echo "NCubeV failed with unexpected exit status ${ncube_status}. See ${log_file}."
      exit "${ncube_status}"
    fi
  fi

  echo "Run ${run_nb}: NCubeV found counterexamples. Converting them for the next round."

  (
    cd "${SCRIPT_DIR}"
    echo "[Run ${run_nb}] NCubeV found unsafe stars; converting JLD to PKL"
    expected="${OUTPUT_DIR}/acc_bigger_polytopes-${run_nb}.pkl"
    julia acc_Ncube_polytope_convert.jl "${OUTPUT_DIR}/acc_bigger_polytopes-${run_nb}-final.jld"
    test -s "${expected}"
    "${PYTHON_BIN}" -c 'import pickle, sys; regions = pickle.load(open(sys.argv[1], "rb")); assert regions, "NCubeV returned only uncertain stars; no certain counterexample region is available for retraining"' "${expected}"

    echo "[Run ${run_nb}] Completed successfully"
  ) >> "${log_file}" 2>&1

  echo "Run ${run_nb} post-processing completed. Retrying supervised retraining."
  run_nb=$((run_nb + 1))
done

echo "Supervised CEGIS loop ended after NCubeV certified run ${run_nb}."
