import transformers
from transformers import BloomForCausalLM
from transformers import BloomTokenizerFast
import torch
import sys
import os

# os.environ['HF_DATASETS_OFFLINE']='1'
# os.environ['TRANSFORMERS_OFFLINE']='1'

model = BloomForCausalLM.from_pretrained("/home/hz0567/data/huggingface_hub/bigscience/bloom-7b1", device_map="auto",torch_dtype="auto")
tokenizer = BloomTokenizerFast.from_pretrained("/home/hz0567/data/huggingface_hub/bigscience/bloom-7b1")
print(model.hf_device_map)

prompt = "Question: who is smarter, Newton or Einstein Answer:"
prompt = [prompt, prompt, prompt, prompt, prompt, prompt, prompt, prompt, prompt, prompt]
result_length = 100
inputs = tokenizer(prompt, return_tensors="pt")

# Greedy Search
print(tokenizer.decode(model.generate(inputs["input_ids"], 
                       max_length=result_length
                      )[0]))

# Beam Search
print(tokenizer.decode(model.generate(inputs["input_ids"],
                       max_length=result_length, 
                       num_beams=2, 
                       no_repeat_ngram_size=2,
                       early_stopping=True
                      )[0]))

# Sampling Top-k + Top-p
print(tokenizer.decode(model.generate(inputs["input_ids"],
                       max_length=result_length, 
                       do_sample=True, 
                       top_k=50, 
                       top_p=0.9
                      )[0]))

# binary search function


