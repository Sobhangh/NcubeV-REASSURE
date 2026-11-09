#!/usr/bin/env bash
set -euo pipefail

echo "Running example: ACC REASSURE"

start_time=$(date +%s)
print_total_runtime() {
  local end_time elapsed
  end_time=$(date +%s)
  elapsed=$((end_time - start_time))
  echo "Total runtime: ${elapsed}s"
}
trap print_total_runtime EXIT

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
OUTPUT_DIR="${1:-${SCRIPT_DIR}/reassure}"
UPPER_BOUND="${UPPER_BOUND:-100}"
MAX_ROUNDS="${MAX_ROUNDS:-100}"
SUPPORT_SHARPNESS="${SUPPORT_SHARPNESS:-10000}"
SUPPORT_MARGIN="${SUPPORT_MARGIN:-1e-5}"
INITIAL_MODEL_PT="${SCRIPT_DIR}/path_RSSR/ppo_acc_bigger_200000_steps.pt"
INITIAL_MODEL_ONNX="${SCRIPT_DIR}/path_RSSR/ppo_acc_bigger_200000_steps.onnx"
INITIAL_POLYTOPES="${SCRIPT_DIR}/path_RSSR/acc_bigger_polytopes.pkl"

source "${SCRIPT_DIR}/env.sh"
PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
JULIA_BIN="${SCRIPT_DIR}/.julia-bin/julia"

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR=$(cd "${OUTPUT_DIR}" && pwd)
LOG_DIR="${OUTPUT_DIR}/logs"
UPPER_BOUND_TAG=$("${PYTHON_BIN}" -c \
  'import sys; x=float(sys.argv[1]); print(int(x) if x.is_integer() else format(x, "g"))' \
  "${UPPER_BOUND}")
mkdir -p "${LOG_DIR}"
test -x "${NCUBEV_BIN}"
test -x "${PYTHON_BIN}"
test -x "${JULIA_BIN}"
test -s "${INITIAL_MODEL_PT}"
test -s "${INITIAL_MODEL_ONNX}"

# NNEnum does not currently accept the exported REASSURE graph.  This mode
# makes the controller available for evaluation without implying certification.
if [[ "${BUILD_ONLY:-0}" == "1" ]]; then
  test -s "${INITIAL_POLYTOPES}"
  echo "Building a REASSURE controller from the existing CE regions."
  (
    cd "${SCRIPT_DIR}"
    PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u acc_REASSURE.py \
      1 "${UPPER_BOUND}" \
      --output-dir "${OUTPUT_DIR}" \
      --initial-model "${INITIAL_MODEL_PT}" \
      --initial-polytopes "${INITIAL_POLYTOPES}" \
      --support-sharpness "${SUPPORT_SHARPNESS}" \
      --support-margin "${SUPPORT_MARGIN}"
  ) > "${LOG_DIR}/reassure_repair_build.log" 2>&1
  echo "Built REASSURE controller in ${OUTPUT_DIR}; formal verification was skipped."
  exit 0
fi

run_ncubev() {
  local model_path=$1
  local result_prefix=$2
  (
    cd "${SCRIPT_DIR}"
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "${NCUBEV_BIN}" \
      NCubeV/test/parsing/examples/acc/formula \
      NCubeV/test/parsing/examples/acc/fixed \
      NCubeV/test/parsing/examples/acc/mapping \
      "${model_path}" "${result_prefix}" --approx 1
  )
}

convert_counterexamples() {
  local round=$1
  local final_jld="${OUTPUT_DIR}/acc_bigger_polytopes-${round}-final.jld"
  local output_pkl="${OUTPUT_DIR}/acc_bigger_polytopes-${round}.pkl"
  test -s "${final_jld}"
  (
    cd "${SCRIPT_DIR}"
    "${JULIA_BIN}" acc_Ncube_polytope_convert.jl "${final_jld}"
  )
  test -s "${output_pkl}"
  "${PYTHON_BIN}" -c \
    'import pickle, sys; regions = pickle.load(open(sys.argv[1], "rb")); assert regions, "NCubeV returned no certain counterexample regions"' \
    "${output_pkl}"
}

echo "Output directory: ${OUTPUT_DIR}"
echo "Initial NCubeV verification"
initial_log="${LOG_DIR}/reassure_repair_run_initial.log"
if run_ncubev \
    "${INITIAL_MODEL_ONNX}" \
    "${OUTPUT_DIR}/acc_bigger_polytopes-0" \
    > "${initial_log}" 2>&1; then
  echo "The original controller is already formally certified; no repair is needed."
  exit 0
else
  status=$?
  if [[ "${status}" -ne 1 ]]; then
    echo "Initial NCubeV verification failed with status ${status}. See ${initial_log}." >&2
    exit "${status}"
  fi
  if ! grep -Eq '^# Unsafe Stars: [1-9][0-9]*' "${initial_log}" ||
      [[ ! -s "${OUTPUT_DIR}/acc_bigger_polytopes-0-final.jld" ]]; then
    echo "Initial NCubeV exited without usable counterexamples; see ${initial_log}." >&2
    exit 1
  fi
fi

echo "Initial NCubeV run found counterexamples; converting them for REASSURE."
convert_counterexamples 0 >> "${initial_log}" 2>&1

round=1
while (( round <= MAX_ROUNDS )); do
  log_file="${LOG_DIR}/reassure_repair_run_${round}_ub_${UPPER_BOUND}.log"
  echo "Starting REASSURE round ${round}. Logs: ${log_file}"

  (
    cd "${SCRIPT_DIR}"
    echo "[Run ${round}] Constructing REASSURE support network"
    PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u acc_REASSURE.py \
      "${round}" "${UPPER_BOUND}" \
      --output-dir "${OUTPUT_DIR}" \
      --initial-model "${INITIAL_MODEL_PT}" \
      --initial-polytopes "${OUTPUT_DIR}/acc_bigger_polytopes-0.pkl" \
      --support-sharpness "${SUPPORT_SHARPNESS}" \
      --support-margin "${SUPPORT_MARGIN}"
  ) > "${log_file}" 2>&1

  model="${OUTPUT_DIR}/ppo_acc_bigger_200000_steps-${UPPER_BOUND_TAG}-${round}.onnx"
  test -s "${model}"
  echo "Round ${round} repair completed. Running NCubeV."
  if run_ncubev \
      "${model}" \
      "${OUTPUT_DIR}/acc_bigger_polytopes-${round}" \
      >> "${log_file}" 2>&1; then
    echo "Round ${round} formally certified by NCubeV. Stopping loop."
    echo "REASSURE CEGIS loop ended after NCubeV certified round ${round}."
    exit 0
  else
    status=$?
    if [[ "${status}" -ne 1 ]]; then
      echo "NCubeV failed with status ${status}. See ${log_file}." >&2
      exit "${status}"
    fi
    if ! grep -Eq '^# Unsafe Stars: [1-9][0-9]*' "${log_file}" ||
        [[ ! -s "${OUTPUT_DIR}/acc_bigger_polytopes-${round}-final.jld" ]]; then
      echo "NCubeV could not verify the REASSURE graph; see ${log_file}." >&2
      exit 1
    fi
  fi

  echo "Round ${round}: NCubeV found counterexamples; converting them for the next round."
  convert_counterexamples "${round}" >> "${log_file}" 2>&1
  round=$((round + 1))
done

echo "REASSURE did not certify within ${MAX_ROUNDS} rounds." >&2
exit 1
