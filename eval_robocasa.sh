#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 MODEL_PATH [additional args passed to a_interact_robocasa_env.py]"
  echo "Examples:"
  echo "  $0 /path/to/checkpoint --num_episodes 5"
  echo "  $0 /path/to/checkpoint --num_episodes 5 --save_video"
  exit 1
fi

MODEL_PATH="$1"
shift || true

PY_SCRIPT="/fsx/kimin/Issac-GR00T-robocasa/eval_robocasa.py"
DATE_PREFIX="$(date +%Y%m%d_%H%M%S)"

envs=(
  "PnPCabToCounter"
  "PnPCounterToCab"
  "TurnOffSinkFaucet"
  "TurnOnSinkFaucet"
  "CloseDoubleDoor"
  "OpenDoubleDoor"
  "CoffeeServeMug"
  "CoffeeSetupMug"
)

for ENV_NAME in "${envs[@]}"; do
  RUN_NAME="${DATE_PREFIX}_${ENV_NAME}"
  echo "Running ${ENV_NAME} -> run_name=${RUN_NAME}"
  python "$PY_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --env_name "$ENV_NAME" \
    "$@"
  echo "Completed ${ENV_NAME}"
  echo
done

echo "All runs completed." 