import transformers
from transformers import BloomForCausalLM
from transformers import BloomTokenizerFast
import torch
import sys
import os

bloom_7b1_path = os.environ.get("BLOOM_7B1_PATH")
model = BloomForCausalLM.from_pretrained(bloom_7b1_path, device_map="auto",torch_dtype="auto") # change this to your path
tokenizer = BloomTokenizerFast.from_pretrained(bloom_7b1_path) # change this to your path
print(model.hf_device_map)

prompt = "Question: who is smarter, Newton or Einstein Answer:" # change this to your prompt
prompt = [prompt, prompt, prompt, prompt, prompt, prompt, prompt, prompt, prompt, prompt] # change this to your list of prompts
result_length = 100
inputs = tokenizer(prompt, return_tensors="pt")

# Greedy Search
print(tokenizer.decode(model.generate(inputs["input_ids"], 
                       max_length=result_length
                      )[0]))




