import os
import time
import requests
import json
from glob import glob
from typing import List, Dict, Any, Optional
import argparse
from tqdm import tqdm
import subprocess
import json
import shlex
import os

from loguru import logger
from common.common_utils import read_json, save_to_json
from llm_infer_directly import load_test_data


def load_demo_dialogs(dataset: str, qtype: Optional[str] = None, entity: Optional[str] = None, fix_4_shot: bool = False) -> List[str]:
    """
    Load demo dialogs from files based on dataset, qtype, and entity settings.
    Similar to _load_demo_dialogs in KBQARunner.
    """
    if fix_4_shot:
        pattern = f"fewshot_demo/{dataset}/dialog-4-shot/*.txt"
    else:
        base = f"fewshot_demo/{dataset}/dialog"
        if entity:
            base += f"-{entity}-entity"
        pattern = base + "/"
        pattern += f"{qtype}-[0-9][0-9].txt" if qtype else "*.txt"

    logger.info(f"Loading demos from: {pattern}")
    demos = []
    for path in glob(pattern):
        with open(path, "r") as f:
            lines = [line for line in f if not line.startswith("#")]
            content = "".join(lines).strip()
            demos.extend(content.split("\n\n"))

    if qtype and not fix_4_shot:
        assert len(demos) == 2, f"Expected 2 demos for qtype '{qtype}', got {len(demos)}"
    logger.info(f"Loaded {len(demos)} demos")
    return demos


def get_tool_desc(dataset: str, qtype: Optional[str] = None) -> tuple:
    """Get the appropriate tool description and db for the dataset."""
    from common.constant import TOOL_DESC_FULL_FB, TOOL_DESC_FULL_KQAPRO, TOOL_DESC_FULL_METAQA
    
    if dataset in ["webqsp", "cwq"]:
        return "fb", TOOL_DESC_FULL_FB
    elif dataset == "kqapro":
        if qtype is None:
            logger.warning("qtype should be provided for kqapro")
        return "kqapro", TOOL_DESC_FULL_KQAPRO
    elif dataset == "metaqa":
        return "metaqa", TOOL_DESC_FULL_METAQA
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def prepare_data(dataset: str, case_num: Optional[int] = None, qtype: Optional[str] = None) -> List[Dict]:
    """Prepare data from dataset, filtering by qtype if specified."""
    data = load_test_data(dataset, case_num=case_num)
    
    if qtype:
        pred_file = f"data_preprocess/{dataset}-classification-prediction.json"
        if not os.path.exists(pred_file):
            logger.warning(f"Missing prediction file: {pred_file}. Not filtering by qtype.")
        else:
            preds = read_json(pred_file)
            id_to_pred = {p["id"]: p["pred_label"] for p in preds}
            data = [d for d in data if id_to_pred.get(d["id"]) == qtype]
            for d in data:
                d["pred_label"] = qtype
    
    return data


def send_to_server(server_url: str, db: str, messages: List[Dict], stop_tokens: List[str] = None, **generation_config) -> Dict:
    """Send messages to the dialog server using curl and return the response."""
    endpoint = f"{server_url}/{db}"
    
    payload = {
        "messages": messages,
        "db": db,
        "stop": stop_tokens,
        **generation_config
    }
    
    # Lưu payload vào file tạm thời (an toàn hơn với JSON phức tạp)
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp:
        json.dump(payload, temp)
        temp_path = temp.name
    
    try:
        # Tạo lệnh curl
        curl_cmd = f'curl -X POST "{endpoint}" -H "Content-Type: application/json" -H "accept: application/json" -d @{temp_path} --silent'
        
        logger.info(f"Executing: {curl_cmd}")
        result = subprocess.run(shlex.split(curl_cmd), capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.error(f"Curl failed with code {result.returncode}: {result.stderr}")
            return {"error": f"Curl error: {result.stderr}", "success": False}
        
        # Parse response
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse response: {result.stdout[:200]}...")
            return {"error": "Invalid JSON response", "success": False}
    except Exception as e:
        logger.error(f"Error executing curl: {e}")
        return {"error": str(e), "success": False}
    finally:
        # Xóa file tạm
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def process_conversation(
    question: str, 
    server_url: str, 
    db: str, 
    tool_demos: str, 
    max_rounds: int = 10,
    **generation_config
) -> List[Dict]:
    """Process a complete conversation with the model."""
    messages = [
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": tool_demos + f"\n\nQ: {question}\nThought:"}
    ]
    
    stop_tokens = ["\nObservation", "\nThought"]
    conversation = []
    
    for round_idx in range(max_rounds):
        response = send_to_server(server_url, db, messages, stop_tokens, **generation_config)
        
        if not response.get("success", False):
            logger.error(f"Error in round {round_idx}: {response.get('error', 'Unknown error')}")
            break
            
        # Extract model output
        model_output = response["choices"][0]["message"]["content"].strip()
        conversation.append({"role": "assistant", "content": model_output})
        messages.append({"role": "assistant", "content": model_output})
        
        # Check if model output contains Done action
        if "Action: Done" in model_output:
            conversation.append({"role": "user", "content": "Stop condition detected."})
            break
            
        # Parse action and get observation
        from tool.action_execution import parse_action
        observation = parse_action(model_output, db=db, execute=True)
        observation = str(observation).strip()
        
        conversation.append({"role": "user", "content": f"Observation: {observation}"})
        messages.append({"role": "user", "content": f"Observation: {observation}"})
    
    return conversation


def main():
    parser = argparse.ArgumentParser(description="Dialog Client for Interactive KBQA")
    parser.add_argument("--server_url", type=str, required=True, help="URL of the dialog server")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use (webqsp, cwq, kqapro, metaqa)")
    parser.add_argument("--case_num", type=int, default=10, help="Number of cases to process")
    parser.add_argument("--qtype", type=str, default=None, help="Question type to filter by")
    parser.add_argument("--entity", type=str, default=None, help="Entity type (None or 'golden')")
    parser.add_argument("--fix_4_shot", action="store_true", help="Use fixed 4-shot demos")
    parser.add_argument("--temperature", type=float, default=0.2, help="Model temperature")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save results")
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU usage on server")
    
    args = parser.parse_args()
    
    # Validate arguments
    assert args.dataset in ["webqsp", "cwq", "kqapro", "metaqa"], f"Unsupported dataset: {args.dataset}"
    if args.entity:
        assert args.entity == "golden", "Entity must be None or 'golden'"
    
    # Get database and tool description
    db, tool_desc = get_tool_desc(args.dataset, args.qtype)
    
    # Load demos
    demos = load_demo_dialogs(args.dataset, args.qtype, args.entity, args.fix_4_shot)

    logger.info(f"Loaded {len(demos)} demo dialogs")
    tool_demos = tool_desc + "\n\n" + "\n\n".join(demos)

    
    # Prepare data
    data = prepare_data(args.dataset, args.case_num, args.qtype)

    logger.info(f"Loaded {len(data)} items from {args.dataset}")
    
    # Create save directory
    if args.save_dir is None:
        name = args.dataset
        if args.qtype:
            name += f"-{args.qtype}"
        if args.entity:
            name += f"-{args.entity}"
        args.save_dir = f"save-dialog-client/{name}"
    
    os.makedirs(args.save_dir, exist_ok=True)

    

    # Process data
    generation_config = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "top_p": 1.0,
        "repetition_penalty": 1.1,
        "num_return_sequences": 1,
        "use_cpu": args.use_cpu
    }
    
    logger.info(f"Processing {len(data)} items from {args.dataset}")

        # Check server connection
    test_url = f"{args.server_url}/{db}/test"
    logger.info(f"Testing server connection at {test_url}")
    try:
        curl_cmd = f'curl -X GET "{test_url}" --silent'
        result = subprocess.run(shlex.split(curl_cmd), capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            try:
                response_data = json.loads(result.stdout)
                logger.info(f"Server status: {response_data}")
            except json.JSONDecodeError:
                logger.warning(f"Server returned non-JSON response: {result.stdout}")
                if "status" in result.stdout and "ok" in result.stdout:
                    logger.info("Server appears to be working despite JSON parsing issue")
                else:
                    raise Exception("Invalid server response")
        else:
            logger.error(f"Curl command failed: {result.stderr}")
            return
    except Exception as e:
        logger.error(f"Cannot connect to server: {e}")
        return
    
    for item in tqdm(data):
        item_id = item["id"]
        save_path = os.path.join(args.save_dir, f"{item_id}.json")
        
        # Skip if already processed
        if os.path.exists(save_path):
            logger.debug(f"Skipping {item_id}, already processed")
            continue
        
        logger.info(f"Processing item {item_id}: {item['question']}")
        
        try:
            conversation = process_conversation(
                question=item["question"],
                server_url=args.server_url,
                db=db,
                tool_demos=tool_demos,
                max_rounds=10,
                **generation_config
            )
            
            result = {
                "id": item_id,
                "question": item["question"],
                "model_name": "server-model",  # The actual model is on the server
                "dialog": conversation,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            save_to_json(result, save_path)
            
        except Exception as e:
            logger.error(f"Error processing item {item_id}: {e}")
    
    logger.info(f"Processing completed. Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()