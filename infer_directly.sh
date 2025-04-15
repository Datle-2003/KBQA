SERVER_URL="https://29de-35-196-216-226.ngrok-free.app"
MODEL_PATH="/kaggle/working/Llama-2-7b-hf/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9/"

python llm_interactive_kbqa.py \
    --dataset kqapro \
    --model_name $MODEL_PATH \
    -qtype Count