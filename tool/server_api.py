import requests
import json

def server_chat(
    prompt=None,
    messages=None,
    model=None,
    temperature=0.7,
    top_p=1,
    n=1,
    stop=None,
    max_tokens=256,
    **kwargs
):
    """Client function to send requests to the FastAPI server instead of OpenAI API"""
    
    server_url = "https://a5ce-34-90-63-121.ngrok-free.app/kqapro" 
    
    # Nếu chỉ có prompt, chuyển thành messages
    if prompt and not messages:
        messages = [{"role": "user", "content": prompt}]
    
    payload = {
        "messages": messages,
        "db": "kqapro",  # or fb, metaqa, etc.
        "stop": stop,
        "max_new_tokens": max_tokens,
        "num_beams": 1,
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "repetition_penalty": 1.0,
        "num_return_sequences": n,
        "use_cpu": True  # Theo yêu cầu của bạn
    }
    
    try:
        response = requests.post(
            server_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=300  # Longer timeout for large models
        )
        
        if response.status_code != 200:
            return {"error": f"Server returned status code {response.status_code}: {response.text}"}
        
        result = response.json()
        
        # Chuyển định dạng của response để giống với response của OpenAI API
        formatted_response = {
            "choices": [
                {
                    "message": {"content": text},
                    "index": i
                } for i, text in enumerate(result.get("choices", [{"message": {"content": ""}}]))
            ],
            "usage": result.get("usage", {"prompt_tokens": 0, "completion_tokens": 0})
        }
        
        return formatted_response
    
    except Exception as e:
        return {"error": str(e)}