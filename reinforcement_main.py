"""
# imports
import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import respond_to_batch

checkpoint =  "Salesforce/codegen-350M-mono"

# get models
model = AutoModelForCausalLMWithValueHead.from_pretrained(checkpoint)
model_ref = create_reference_model(model)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #model.resize_token_embeddings(len(tokenizer))


# initialize trainer
ppo_config = PPOConfig(
    batch_size=1,
)

# encode a query
query_txt = "This morning I went to the "
query_tensor = tokenizer.encode(query_txt, return_tensors="pt")

# get model response
response_tensor  = respond_to_batch(model, query_tensor)

# create a ppo trainer
ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer)

# define a reward for response
# (this could be any reward such as human feedback or output from another model)
reward = [torch.tensor(1.0)]

# train model for one step with ppo
train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
print("Printing Training stats...", train_stats)
"""


import torch
from transformers import GPT2Tokenizer
checkpoint =  "Salesforce/codegen-350M-mono"
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer


# 1. load a pretrained model
model = AutoModelForCausalLMWithValueHead.from_pretrained(checkpoint)
model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(checkpoint)
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token

# 2. initialize trainer
ppo_config = {"batch_size": 1}
config = PPOConfig(**ppo_config)
ppo_trainer = PPOTrainer(config, model, model_ref, tokenizer)

# 3. encode a query
query_txt = "This morning I went to the "
query_tensor = tokenizer.encode(query_txt, return_tensors="pt").to(model.pretrained_model.device)

# 4. generate model response
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 20,
}
response_tensor = ppo_trainer.generate([item for item in query_tensor], return_prompt=False, **generation_kwargs)
response_txt = tokenizer.decode(response_tensor[0])
print(response_txt)

# 5. define a reward for response
# (this could be any reward such as human feedback or output from another model)
reward = [torch.tensor(1.0, device=model.pretrained_model.device)]

# 6. train model with ppo
train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)