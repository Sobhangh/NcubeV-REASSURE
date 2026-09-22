#set -euo pipefail

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
LOG_DIR="${SCRIPT_DIR}/supervised/logs"

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
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./NCubeV/deps/NCubeV/bin/NCubeV NCubeV/test/parsing/examples/acc/formula NCubeV/test/parsing/examples/acc/fixed NCubeV/test/parsing/examples/acc/mapping "${SCRIPT_DIR}/supervised/${1}.onnx" "${SCRIPT_DIR}/supervised/${2}" --approx 1
}

run_nb=1
while true; do
  log_file="${LOG_DIR}/supervised_repair_run_${run_nb}.log"
  echo "Starting run ${run_nb}. Logs: ${log_file}"

  if (
    cd "${SCRIPT_DIR}"
    echo "[Run ${run_nb}] Starting supervised retraining"
    "${PYTHON_BIN}" acc_supervised_retrain.py "${run_nb}"
  ) > "${log_file}" 2>&1; then
    echo "Run ${run_nb} reached zero crashes. Stopping loop."
    break
  fi

  echo "Run ${run_nb} did not reach zero crashes. Running NCubeV + conversion before retry."

  (
    cd "${SCRIPT_DIR}"
    echo "[Run ${run_nb}] Starting NCubeV verification"
    run_ncubev "ppo_acc_bigger_200000_steps-${run_nb}" "acc_bigger_polytopes-${run_nb}"

    echo "[Run ${run_nb}] Converting JLD to PKL"
    julia acc_Ncube_polytope_convert.jl "supervised/acc_bigger_polytopes-${run_nb}-final.jld"

    echo "[Run ${run_nb}] Completed successfully"
  ) >> "${log_file}" 2>&1

  echo "Run ${run_nb} post-processing completed. Retrying supervised retraining."
  run_nb=$((run_nb + 1))
done

echo "Supervised loop ended after successful retraining run ${run_nb}."