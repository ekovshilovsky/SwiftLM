import json, glob, os
from transformers import AutoTokenizer

model_id = "mlx-community/gemma-4-e4b-it-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "user", "content": "Hey! What is the capital of France?"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)
