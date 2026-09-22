#set -euo pipefail

echo "Running example: ACC REASSURE"
echo "BEWARE: Before running this you need to build NCubeV and have the REASSURE dependencies available"

start_time=$(date +%s)

print_total_runtime() {
  local end_time elapsed
  end_time=$(date +%s)
  elapsed=$((end_time - start_time))
  echo "Total runtime: ${elapsed}s"
}

trap print_total_runtime EXIT

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
LOG_DIR="${SCRIPT_DIR}/reassure/logs"
UPPER_BOUND=100

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
else
  echo "Error: neither 'python' nor 'python3' is available in PATH."
  exit 1
fi

if ! "${PYTHON_BIN}" -c "import polytope" >/dev/null 2>&1; then
  echo "polytope is not installed for ${PYTHON_BIN}; installing with pip"
  "${PYTHON_BIN}" -m pip install --break-system-packages polytope
fi

mkdir -p "${LOG_DIR}"

run_ncubev () {
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./NCubeV/deps/NCubeV/bin/NCubeV NCubeV/test/parsing/examples/acc/formula NCubeV/test/parsing/examples/acc/fixed NCubeV/test/parsing/examples/acc/mapping "${SCRIPT_DIR}/path_RSSR/${1}.onnx" "${SCRIPT_DIR}/path_RSSR/${2}.jld" --approx 1
}

echo "Initial Run Starting NCubeV verification"
log_file="${LOG_DIR}/reassure_repair_run_initial.log"
run_ncubev "ppo_acc_bigger_200000_steps" "acc_bigger_polytopes" > "${log_file}" 2>&1 || {
    echo "Initial run failed. Full log from ${log_file}:"
    cat "${log_file}"
    exit 1
  }

echo "Initial Run Converting JLD to PKL"
julia acc_Ncube_polytope_convert.jl "path_RSSR/acc_bigger_polytopes.jld" > "${log_file}" 2>&1 || {
    echo "Initial run failed. Full log from ${log_file}:"
    cat "${log_file}"
    exit 1
  }

for i in $(seq 1 2); do
  log_file="${LOG_DIR}/reassure_repair_run_${i}_ub_${UPPER_BOUND}.log"
  echo "Starting run ${i} with upper bound ${UPPER_BOUND}. Logs: ${log_file}"

  (
    cd "${SCRIPT_DIR}"
    echo "[Run ${i}] Starting REASSURE repair"
    "${PYTHON_BIN}" acc_REASSURE.py "${i}" "${UPPER_BOUND}"

    echo "[Run ${i}] Starting NCubeV verification"
    run_ncubev "ppo_acc_bigger_200000_steps-${UPPER_BOUND}-${i}" "acc_bigger_polytopes-${i}"

    echo "[Run ${i}] Converting JLD to PKL"
    julia acc_Ncube_polytope_convert.jl "path_RSSR/acc_bigger_polytopes-${i}.jld"

    echo "[Run ${i}] Completed successfully"
  ) > "${log_file}" 2>&1 || {
    echo "Run ${i} failed. Full log from ${log_file}:"
    cat "${log_file}"
    exit 1
  }

  echo "Run ${i} completed successfully."
done