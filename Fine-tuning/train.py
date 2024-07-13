from datasets import load_dataset

train_dataset = load_dataset('json', data_files='train.jsonl', split='train')
eval_dataset = load_dataset('json', data_files='validation.jsonl', split='train')

print(train_dataset)
print(eval_dataset)

from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

# fsdp_plugin = FullyShardedDataParallelPlugin(
#     state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
#     optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
# )

# accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

import wandb, os
wandb.login()

wandb_project = "journal-finetune"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

def formatting_func(example):
    text = f"### Question: {example['question']}\n ### Answer: {example['answer']}"
    return text

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

def generate_and_tokenize_prompt(prompt):
    return tokenizer(formatting_func(prompt))

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

import matplotlib.pyplot as plt

def plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset):
    lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
    lengths += [len(x['input_ids']) for x in tokenized_val_dataset]
    print(len(lengths))

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Length of input_ids')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of input_ids')
    plt.savefig("lengths_histogram.png")

plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset)

max_length = 2875 # This was an appropriate max length for my dataset

def generate_and_tokenize_prompt2(prompt):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result
tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt2)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt2)
#print(tokenized_train_dataset[1]['input_ids'])
plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset)

eval_prompt = "Given the table title \" Roman Frontier Defenses along the Cumbrian Coast: A List of Sites and Distances  \", summary \" This table provides the names and distances (in meters) of various sites related to the Roman frontier defenses along the Cumbrian coast. The list includes tower sites, milefortlets, and other significant locations such as golf courses and fortifications. Notable sites include Herd Hill North (tower 3b), Blitterlees (milefortlet 12), Sea Brows (milefortlet 23), Low Mire (milefortlet 20), and Drumburgh Roman fort and Hadrian's Wall between Burgh Marsh and Westfield House. The distances are given relative to various landmarks, such as farms, houses, and wash sites.\", row \"col: | col0 | col1 |   row 1: | Herd Hill North (tower 3b), 175m north east of the sheep wash, partof the Roman frontier defences along the Cumbrian coast | 0.07063 |  \", along with referent entity candidates ID | NAME | DESCRIPTION | TYPEQ64811792 | herd hill north , 175m north east of the sheep wash, part of the roman frontier defences along the cumbrian coast | part of a world heritage site in the united kingdom | [{id: Q839954, name: archaeological site}],Q17673645 | herd hill north , 175m north east of the sheep wash, part of the roman frontier defences along the cumbrian coast | nan | [{id: Q839954, name: archaeological site}],Q17662948 | brownrigg milefortlet 22, 800m north east of the cemetery chapel, part of the roman frontier defences along the cumbrian coast | fortification in crosscanonby, allerdale, england, uk | [{id: Q57821, name: fortification}],Q64811756 | brownrigg milefortlet 22, 800m north east of the cemetery chapel, part of the roman frontier defences along the cumbrian coast | part of a world heritage site in the united kingdom | [{id: Q839954, name: archaeological site}],Q64811771 | swarthy hill north tower 20b, 460m south west of blue dial, part of the roman frontier defences along the cumbrian coast | part of a world heritage site in the united kingdom | [{id: Q839954, name: archaeological site}],Q17664297 | swarthy hill north tower 20b, 460m south west of blue dial, part of the roman frontier defences along the cumbrian coast | nan | [{id: Q839954, name: archaeological site}],Q64811797 | herd hill and associated parallel banks and ditches, part of the roman frontier defences along the cumbrian coast | part of a world heritage site in the united kingdom | [{id: Q839954, name: archaeological site}],Q64811775 | brownrigg north tower 21b, 830m north west of canonby hall, part of the roman frontier defences along the cumbrian coast | part of a world heritage site in the united kingdom | [{id: Q839954, name: archaeological site}],Q64811846 | cardurnock and earlier ditch system and patrol road, part of the roman frontier defences along the cumbrian coast | part of a world heritage site in the united kingdom | [{id: Q839954, name: archaeological site}],Q17664307 | brownrigg north tower 21b, 830m north west of canonby hall, part of the roman frontier defences along the cumbrian coast | nan | [{id: Q839954, name: archaeological site}],Q17677624 | cardurnock and earlier ditch system and patrol road, part of the roman frontier defences along the cumbrian coast | nan | [{id: Q839954, name: archaeological site}],Q17673565 | low mire 50m north of heather bank, part of the roman frontier defences along the cumbrian coast | fortification in oughterside and allerby, allerdale, england, uk | [{id: Q57821, name: fortification}],Q64811784 | low mire 50m north of heather bank, part of the roman frontier defences along the cumbrian coast | part of a world heritage site in the united kingdom | [{id: Q839954, name: archaeological site}],Q17668070 | maryport golf course tower 22a, 350m north of the cemetery chapel, part of the roman frontier defences along the cumbrian coast | nan | [{id: Q839954, name: archaeological site}],Q64811778 | maryport golf course tower 22a, 350m north of the cemetery chapel, part of the roman frontier defences along the cumbrian coast | part of a world heritage site in the united kingdom | [{id: Q839954, name: archaeological site}],Q17662954 | bank mill tower 15a, 250m north west of belmont house, part of the roman frontier defences along the cumbrian coast | mill building in holme st cuthbert, allerdale, england, uk | [{id: Q56822897, name: mill building}],Q17662950 | wolsty north tower 13a, 500m south west of wolsty farm, part of the roman frontier defences along the cumbrian coast | nan | [{id: Q839954, name: archaeological site}],Q64811760 | wolsty north tower 13a, 500m south west of wolsty farm, part of the roman frontier defences along the cumbrian coast | part of a world heritage site in the united kingdom | [{id: Q839954, name: archaeological site}],Q64811765 | bank mill tower 15a, 250m north west of belmont house, part of the roman frontier defences along the cumbrian coast | part of a world heritage site in the united kingdom | [{id: Q839954, name: archaeological site}],Q17662936 | silloth golf course tower 12b, 410m north west of heatherbank, part of the roman frontier defences along the cumbrian coast | part of a world heritage site in the united kingdom | [{id: Q839954, name: archaeological site}], and wikidata \"No good Wikidata Search Result was found\", identify the correct referent entity candidate for Herd Hill North (tower 3b), 175m north east of the sheep wash, partof the Roman frontier defences along the Cumbrian coast.: ### Answer:",

# Init an eval tokenizer that doesn't add padding or eos token
eval_tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_bos_token=True,
)

model_input = eval_tokenizer(eval_prompt, return_tensors="pt")

model.eval()

with torch.no_grad():
    print(eval_tokenizer.decode(model.generate(**model_input, max_new_tokens=256, repetition_penalty=1.15)[0], skip_special_tokens=True))

from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)