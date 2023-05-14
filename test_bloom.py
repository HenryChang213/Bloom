import transformers
from transformers import BloomForCausalLM
from transformers import BloomTokenizerFast
import torch
import sys

# model = BloomForCausalLM.from_pretrained("bigscience/bloom-1b3")
# tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-1b3")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom", device_map="auto",torch_dtype="auto")
# model = BloomForCausalLM.from_pretrained("bigscience/bloom-1b1", device_map="auto",torch_dtype="auto")
# tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-1b1")
print(model.hf_device_map)
sys.stdout.flush()
exit()
prompt = "Question: who is smarter, Newton or Einstein Answer:"
result_length = 50
inputs = tokenizer(prompt, return_tensors="pt")
print(inputs)
sys.stdout.flush()


start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for i in range(100):
    # Greedy Search
    start.record()
    print(tokenizer.decode(model.generate(inputs["input_ids"].to('cuda'), 
                        max_length=result_length
                        )[0]))
    end.record()

    torch.cuda.synchronize()

    print(start.elapsed_time(end))
    sys.stdout.flush()


# # Beam Search
# print(tokenizer.decode(model.generate(inputs["input_ids"],
#                        max_length=result_length, 
#                        num_beams=2, 
#                        no_repeat_ngram_size=2,
#                        early_stopping=True
#                       )[0]))

# # Sampling Top-k + Top-p
# print(tokenizer.decode(model.generate(inputs["input_ids"],
#                        max_length=result_length, 
#                        do_sample=True, 
#                        top_k=50, 
#                        top_p=0.9
#                       )[0]))