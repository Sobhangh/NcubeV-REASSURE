set -euo pipefail

echo "Running example: ACC"
echo "BEWARE: Before running this you need to build NCubeV"

start_time=$(date +%s)

print_total_runtime() {
  local end_time elapsed
  end_time=$(date +%s)
  elapsed=$((end_time - start_time))
  echo "Total runtime: ${elapsed}s"
}

trap print_total_runtime EXIT

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
LOG_DIR="${SCRIPT_DIR}/retrain/logs"

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

#cd $SCRIPT_DIR/../../

run_ncubev () {
  #mkdir -p experiments/acc/${1}
  # We are now running an experiment.
  # To this end, we call ./deps/NCubeV/bin/NCubeV with the following arguments:
  # test/parsing/examples/acc/formula   -> Formula to verify
  # test/parsing/examples/acc/fixed     -> Fixed variables in the formula
  # test/parsing/examples/acc/mapping   -> Mapping of the variables to the network inputs/outputs
  # test/networks/${1}.onnx             -> ONNX file of the network
  # experiments/acc/${1}/results-approx-${2} -> Prefix of the output files (long runs store intermediate results to save RAM)
  # --approx ${2}                       -> Approximation level \in {1,2,3}
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./NCubeV/deps/NCubeV/bin/NCubeV NCubeV/test/parsing/examples/acc/formula NCubeV/test/parsing/examples/acc/fixed NCubeV/test/parsing/examples/acc/mapping retrain/${1}.onnx retrain/${2}.jld --approx 1
}

for i in $(seq 1 10); do
  log_file="${LOG_DIR}/repair_loop_run_${i}.log"
  echo "Starting run ${i}. Logs: ${log_file}"

  (
    cd "${SCRIPT_DIR}"
    echo "[Run ${i}] Starting PPO retraining"
    "${PYTHON_BIN}" PPO_ACC-Retrain-0.1.py "${i}"

    echo "[Run ${i}] Starting NCubeV verification"
    run_ncubev "ppo_acc_bigger_200000_steps-RETRAIN-${i}" "acc_bigger_polytopes-${i}"

    echo "[Run ${i}] Converting JLD to PKL"
    julia acc_Ncube_polytope_convert.jl "retrain/acc_bigger_polytopes-${i}.jld"

    echo "[Run ${i}] Completed successfully"
  ) > "${log_file}" 2>&1 || {
    echo "Run ${i} failed. Full log from ${log_file}:"
    cat "${log_file}"
    exit 1
  }

  echo "Run ${i} completed successfully."
done