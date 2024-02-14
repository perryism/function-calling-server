from transformers import AutoModelForCausalLM
import sys

model_name = sys.argv[1]

model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True
    )
