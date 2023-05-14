from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
# model = AutoModelForCausalLM.from_pretrained("bigscience/bloom")
# tokenizer.save_pretrained("/home/hz0567/data/huggingface_hub/bigscience/bloom")
# model.save_pretrained("/home/hz0567/data/huggingface_hub/bigscience/bloom")

# tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")
# model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-7b1")
# tokenizer.save_pretrained("/home/hz0567/data/huggingface_hub/bigscience/bloom-7b1")
# model.save_pretrained("/home/hz0567/data/huggingface_hub/bigscience/bloom-7b1")

tokenizer = AutoTokenizer.from_pretrained("microsoft/bloom-deepspeed-inference-int8")
model = AutoModelForCausalLM.from_pretrained("microsoft/bloom-deepspeed-inference-int8")
tokenizer.save_pretrained("/home/hz0567/data/huggingface_hub/microsoft/bloom-deepspeed-inference-int8")
model.save_pretrained("/home/hz0567/data/huggingface_hub/microsoft/bloom-deepspeed-inference-int8")

# from huggingface_hub import hf_hub_download
# hf_hub_download(repo_id="microsoft/bloom-deepspeed-inference-int8", filename="config.json", cache_dir="/home/hz0567/data/huggingface_hub/microsoft/bloom-deepspeed-inference-int8")


