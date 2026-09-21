set -euo pipefail

echo "Running example: ACC SUPERVISED"
echo "BEWARE: Before running this you need to build NCubeV and have the supervised dependencies available"

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
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./NCubeV/deps/NCubeV/bin/NCubeV NCubeV/test/parsing/examples/acc/formula NCubeV/test/parsing/examples/acc/fixed NCubeV/test/parsing/examples/acc/mapping "${SCRIPT_DIR}/supervised/${1}.onnx" "${SCRIPT_DIR}/supervised/${2}.jld" --approx 1
}

for i in $(seq 1 2); do
  log_file="${LOG_DIR}/supervised_repair_run_${i}.log"
  echo "Starting run ${i}. Logs: ${log_file}"

  (
    cd "${SCRIPT_DIR}"
    echo "[Run ${i}] Starting supervised retraining"
    "${PYTHON_BIN}" acc_supervised_retrain.py "${i}"

    echo "[Run ${i}] Starting NCubeV verification"
    run_ncubev "ppo_acc_bigger_200000_steps-${i}" "acc_bigger_polytopes-${i}"

    echo "[Run ${i}] Converting JLD to PKL"
    julia acc_Ncube_polytope_convert.jl "supervised/acc_bigger_polytopes-${i}.jld"

    echo "[Run ${i}] Completed successfully"
  ) > "${log_file}" 2>&1 || {
    echo "Run ${i} failed. Full log from ${log_file}:"
    cat "${log_file}"
    exit 1
  }

  echo "Run ${i} completed successfully."
done