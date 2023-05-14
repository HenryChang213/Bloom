from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")
# model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-7b1")
# tokenizer.save_pretrained("/home/hz0567/data/huggingface/hub/bigscience/bloom-7b1") # change this to your path
# model.save_pretrained("/home/hz0567/data/huggingface/hub/bigscience/bloom-7b1") # change this to your path

# tokenizer = AutoTokenizer.from_pretrained("microsoft/bloom-deepspeed-inference-int8")
# config = AutoConfig.from_pretrained("microsoft/bloom-deepspeed-inference-int8")
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("microsoft/bloom-deepspeed-inference-int8")
model = AutoModel.from_pretrained("microsoft/bloom-deepspeed-inference-int8")
tokenizer.save_pretrained("/home/hz0567/data/huggingface/hub/microsoft/bloom-deepspeed-inference-int8")
model.save_pretrained("/home/hz0567/data/huggingface/hub/microsoft/bloom-deepspeed-inference-int8")

# from huggingface_hub import hf_hub_download
# hf_hub_download(repo_id="microsoft/bloom-deepspeed-inference-int8", filename="config.json", cache_dir="/home/hz0567/data/huggingface_hub/microsoft/bloom-deepspeed-inference-int8")


