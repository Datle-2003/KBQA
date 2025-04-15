import asyncio
import time
import uuid
from traceback import print_exc
from typing import List, Optional, Union, Dict, Any

import fire
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
import torch

from predict.dialog_predictor import predictor_history

class Message(BaseModel):
    role: str
    content: str

    def json(self):
        return {"role": self.role, "content": self.content}


class InputData(BaseModel):
    messages: List[Message]
    db: str
    stop: List[str] = None

    # transformer config
    max_new_tokens: int = 768
    num_beams: int = 1
    do_sample: bool = True
    top_p: float = 1.0
    temperature: float = 1.0
    repetition_penalty: float = 1.0
    num_return_sequences: int = 1
    use_cpu: bool = False 

    def json(self):
        return {
            "messages": [m.json() for m in self.messages],
            "db": self.db,
            "stop": self.stop,
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "do_sample": self.do_sample,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "num_return_sequences": self.num_return_sequences,
            "use_cpu": self.use_cpu,
        }

def start_api(model_name_or_path, use_vllm=False, use_cpu=False, db="fb", port=18000):
    """
    Return
        - response["choices"][0]["message"]["content"]
        - response["usage"]["prompt_tokens"]
        - response["usage"]["completion_tokens"]
    """
    assert db in ["fb", "kqapro", "metaqa", "common"], "db must be fb or kqapro or metaqa"

    cuda_available = torch.cuda.is_available() and not use_cpu
    print("Model path:", model_name_or_path)
    print("Use vLLM:", use_vllm)
    print("Use CPU:", use_cpu)
    print("CUDA available:", cuda_available)
    print("Port:", port)

    app = FastAPI()

    llm = predictor_history(model_name_or_path, db, use_vllm=use_vllm, use_cpu=use_cpu)

    @app.post(f"/{db}")
    async def _chat(data: InputData):
        cpu_mode = data.use_cpu or use_cpu
        
        gen_config = {
            "max_new_tokens": data.max_new_tokens,
            "num_beams": data.num_beams,
            "do_sample": data.do_sample,
            "top_p": data.top_p,
            "temperature": data.temperature,
            "repetition_penalty": data.repetition_penalty,
            "num_return_sequences": data.num_return_sequences,
            "use_cpu": cpu_mode,
        }
        messages = data.model_dump()["messages"]

        start_time = time.time()
        try:
            result = llm(messages, stop=data.stop, **gen_config)
            success = True
        except Exception as e:
            print_exc()
            result = {"error": str(e)}
            success = False
        end_time = time.time()
        response = {
            **result,
            "success": success,
            "time_cost": end_time - start_time,
        }
        return response

    # Test endpoint
    @app.get(f"/{db}/test")
    async def _test():
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        hardware_info = "CPU mode" if use_cpu else "CUDA available" if cuda_available else "CPU mode (CUDA unavailable)"
        return {
            "status": "ok", 
            "start_time": current_time, 
            "db": db,
            "hardware": hardware_info,
            "model": model_name_or_path,
            "use_vllm": use_vllm
        }

    config = uvicorn.Config(app=app, host="0.0.0.0", port=port, workers=1 if use_cpu else 4)
    server = uvicorn.Server(config)

    asyncio.run(server.serve())


if __name__ == "__main__":
    fire.Fire(start_api)


    