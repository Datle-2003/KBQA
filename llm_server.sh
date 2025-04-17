MODEL_PATH="LLMs/TinyLlama/TinyLlama-1.1B-Chat-v1.0/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/"
DB="kqapro" 
PORT=18210

# MODEL_PATH="LLMs/Llama/Llama-2-7b-hf/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9/"
# PORT=18200

echo "Starting CPU-based dialog server for $MODEL_PATH on port $PORT"
python api/api_llm_dialog_server.py \
    --model_name_or_path="$MODEL_PATH" \
    --db="$DB" \
    --port="$PORT" \
    --use_cpu=True


