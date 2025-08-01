#!/bin/bash
#set -e

ENV_NAME="rodsi_eval_env"
RESULTS_DIR="results"
CUSTOM_EVALUATION_SCRIPT="../utils/custom_evaluation.py"


echo "-- Setting up MAMBA Env: $ENV_NAME --"
eval "$(mamba shell hook --shell bash)"
mamba activate $ENV_NAME

echo "STARTED: $(date)"
mkdir -p $RESULTS_DIR


MODEL_LIST=(
  "Qwen/Qwen3-1.7B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  "agentica-org/DeepScaleR-1.5B-Preview"
  "RUC-AIBOX/STILL-3-1.5B-preview"
)

TASK_LIST=(
  "aime24"
  "math_500"
  "gpqa:diamond"
  "aime25"
  "amc23"
  "minerva"
)

echo "-- Starting Eval loop --"

for MODEL in "${MODEL_LIST[@]}"; do

  if [[ $MODEL == "Qwen/Qwen3-1.7B" ]]; then
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,data_parallel_size=1,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
  else
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=1,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
  fi

  for TASK in "${TASK_LIST[@]}"; do
    OUTPUT_PATH="${RESULTS_DIR}/${MODEL//\//_}/${TASK}"

    echo "-------------------------"
    echo "Currently, Evaluation Running of $MODEL on $TASK"

    lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
          --custom-tasks $CUSTOM_EVALUATION_SCRIPT \
          --use-chat-template \
          --output-dir "$OUTPUT_PATH"

    echo "DONE, Results saved to: $OUTPUT_PATH"

  done
done

echo "--- ALL DONE ---"
echo "ENDED: $(date)"

mamba deactivate



