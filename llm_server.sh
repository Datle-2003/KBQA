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


# curl -X POST http://localhost:18210/kqapro \
#   -H "Content-Type: application/json" \
#   -d '{
#     "messages": [
#       {"role": "user", "content": "Who is the president of the United States?"}
#     ],
#     "db": "kqapro",
#     "use_cpu": true,
#     "max_new_tokens": 256,
#     "do_sample": false, 
#     "num_return_sequences": 1
#   }'


# curl -X POST http://localhost:18210/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "LLMs/TinyLlama/TinyLlama-1.1B-Chat-v1.0/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/",
#     "messages": [
#       {"role": "user", "content": "Who is the president of the United States?"}
#     ],
#     "use_cpu": true,
#     "temperature": 0,
#     "max_tokens": 256,
#     "n": 1
#   }'