import uvicorn
import argparse
import uuid

from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM

from server_types import ChatCompletion, ChatInput, Choice
from inference import generate_message
import logging, sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Glaive Function API")

@app.post("/vertexai")
async def vertexai(request: Request):
    data = await request.json()
    instances = data["instances"]
    chat_input = ChatInput.parse_obj(instances[0])
    prediction = await chat_endpoint(chat_input)
    return { "predictions": [prediction.dict()] }

@app.post("/v1/chat/completions", response_model=ChatCompletion)
async def chat_endpoint(chat_input: ChatInput):
    request_id = str(uuid.uuid4())
    response_message = try_generate_message(
        messages=chat_input.messages,
        functions=chat_input.functions,
        temperature=chat_input.temperature,
        model=model,  # type: ignore
        tokenizer=tokenizer,
    )

    return ChatCompletion(
        id=request_id, choices=[Choice.from_message(response_message)]
    )

from demjson3 import JSONDecodeError

def try_generate_message(messages, functions, temperature, model, tokenizer, attempt=3):
    try:
        return generate_message(messages, functions, temperature, model, tokenizer)
    except JSONDecodeError as e:
        if attempt > 0:
            logging.warning(f"Failed to generate message: {e}. Retrying {attempt} more times.")
            return try_generate_message(messages, functions, temperature, model, tokenizer, attempt-1)
        else:
            logger.error(f"Failed to generate message: {e}")
            raise e

@app.get("/ping")
async def ping():
    return "pong"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Glaive Function API")
    parser.add_argument(
        "--model",
        type=str,
        default="glaiveai/glaive-function-calling-v2-small",
        help="Model name",
    )
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)

    uvicorn.run(app, host="0.0.0.0", port=8000)