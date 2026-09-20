echo "Running example: ACC REASSURE"
echo "BEWARE: Before running this you need to build NCubeV and have the REASSURE dependencies available"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
LOG_DIR="${SCRIPT_DIR}/reassure/logs"
UPPER_BOUND=100

mkdir -p "${LOG_DIR}"

run_ncubev () {
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./NCubeV/deps/NCubeV/bin/NCubeV NCubeV/test/parsing/examples/acc/formula NCubeV/test/parsing/examples/acc/fixed NCubeV/test/parsing/examples/acc/mapping "${SCRIPT_DIR}/path_RSSR/${1}.onnx" "${SCRIPT_DIR}/path_RSSR/${2}.jld" --approx 1
}

for i in $(seq 1 2); do
  log_file="${LOG_DIR}/reassure_repair_run_${i}_ub_${UPPER_BOUND}.log"
  echo "Starting run ${i} with upper bound ${UPPER_BOUND}. Logs: ${log_file}"

  if (
    cd "${SCRIPT_DIR}"
    echo "[Run ${i}] Starting REASSURE repair"
    python acc_REASSURE.py "${i}" "${UPPER_BOUND}"

    echo "[Run ${i}] Starting NCubeV verification"
    run_ncubev "ppo_acc_bigger_200000_steps-${UPPER_BOUND}-${i}" "acc_bigger_polytopes-${i}"

    echo "[Run ${i}] Converting JLD to PKL"
    julia acc_Ncube_polytope_convert.jl "path_RSSR/acc_bigger_polytopes-${i}.jld"

    echo "[Run ${i}] Completed successfully"
  ) > "${log_file}" 2>&1; then
    echo "Run ${i} completed successfully."
  else
    echo "Run ${i} failed. Check ${log_file}"
  fi
done