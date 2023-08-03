from transformers import AutoTokenizer, AutoModelForCausalLM, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
#from optimum.bettertransformer import BetterTransformer
import gc
import pandas as pd
from torchmetrics import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from pprint import pprint
from evaluate import load
import evaluate


import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, AutoModelForSeq2SeqLM, T5ForConditionalGeneration

import torch


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score






print("---------------------------- DONE PIAL ----------------------------")


"""
Not working ...
from evaluate import evaluator

import evaluate

module = evaluate.load("dvitel/codebleu")
src = 'class AcidicSwampOoze(MinionCard):§    def __init__(self):§        super().__init__("Acidic Swamp Ooze", 2, CHARACTER_CLASS.ALL, CARD_RARITY.COMMON, battlecry=Battlecry(Destroy(), WeaponSelector(EnemyPlayer())))§§    def create_minion(self, player):§        return Minion(3, 2)§'
tgt = 'class AcidSwampOoze(MinionCard):§    def __init__(self):§        super().__init__("Acidic Swamp Ooze", 2, CHARACTER_CLASS.ALL, CARD_RARITY.COMMON, battlecry=Battlecry(Destroy(), WeaponSelector(EnemyPlayer())))§§    def create_minion(self, player):§        return Minion(3, 2)§'
src = src.replace("§","\n")
tgt = tgt.replace("§","\n")
res = module.compute(predictions = [tgt], references = [[src]])
print(res)
"""


#################################  Running Evaluation Function ######################################

IGNORE_INDEX = 0
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


trained_model = "results_codegen2_3_7B_classification_train"

# device_map="auto"
# Removed device_map = "auto" for santacoder model
model = AutoModelForCausalLM.from_pretrained(trained_model, device_map="auto", trust_remote_code=True)

# For T5 Type models
#model = T5ForConditionalGeneration.from_pretrained(trained_model, device_map="auto",trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(trained_model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
total_params = sum(
	param.numel() for param in model.parameters()
)
print("Total params ...", total_params)
#model.to(device)

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)



def read_data(data_path):
    logging.warning("Loading data...")
    list_data_dict = utils.jload(data_path)

    logging.warning("Formatting inputs...")
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    sources = [
        prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        for example in list_data_dict
    ]
    targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

    #logging.warning("Tokenizing inputs... This may take some time...")

    #print("Printing an element of sources ...", sources[0])
    #print("Printing a target ...", targets[0])

    data_dict = preprocess(sources, targets, tokenizer)
    return sources, targets



def calc_scores():

    file_path = "andrew/instruct-classification-test.json"
    #df = pd.read_csv(file_path)

    total = counter = 50


    blue_score = 0
    gblue = 0
    rouge = 0
    false_counter = 0


    sources, targets = read_data(file_path)

    print(sources[0])
    print(targets[0])

    print(len(sources) == len(targets))



    #print("Input ...", tokenizer.batch_decode(data_dict["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
    #print("Label ...", tokenizer.batch_decode(data_dict["labels"], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

    #return
    y_true = []
    y_pred = []
    generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    for i in range(len(sources)):


        """
        line = tokenizer.batch_decode(line, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        print("Original input line ", line)
        line = ' '.join([str(elem) for elem in line])
        input, output = line.split(" ###  Response :")



        #print("length of prompt", prompt)
        inputs = tokenizer(input, return_tensors="pt", max_length=512).to(device)
        #print("Length of input token", len(inputs))
        # Generate
        # Next 2 are good lines
        #generate_ids = model.generate(inputs.input_ids, max_new_tokens=512, num_beams=10, do_sample=True, early_stopping=True, pad_token_id=tokenizer.eos_token_id)
        #model_output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        #original_output = row["func_after"].strip()
        """
        
        response = generator(sources[i], max_length=1024, do_sample=False)[0]['generated_text']

        #output = bytes(output, "utf-8").decode("unicode_escape")

        original_input = sources[i] #bytes(sources[i], "utf-8").decode("unicode_escape")
        model_output = response#.split("<<")[0][-1] #bytes(response, "utf-8").decode("unicode_escape")
        original_output = targets[i]#.split("<")[0][0] #bytes(targets[i], "utf-8").decode("unicode_escape")
        
        print("----------------------------------------------------------")
        #print("Original Input \n", original_input)
        print("Output from model\n", model_output)
        print("Original Output\n", original_output)
        print("----------------------------------------------------------")
        
        

        try:
            # do something
            y_pred.append(int(model_output))
            y_true.append(int(original_output))
            print(y_true, y_pred)
        except Exception as e:
            pass
        

         # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        # Calculate precision
        precision = precision_score(y_true, y_pred)
        # Calculate recall
        recall = recall_score(y_true, y_pred)
        # Calculate f1 score
        f1 = f1_score(y_true, y_pred)
        print("Accuracy, Precision, Recall, F1", accuracy, precision, recall, f1)
        

        counter -= 1
        if counter == 0:
            break

    #print("Avg. Blue = ", blue_score / counter)
    #print
    #print("Avg Rouge = ", rouge / total)
    #print("Total Blue ...", gblue)
    #print("Avg GBLUE ", gblue / total)

    #print("False Counter ...", false_counter)


    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    # Calculate precision
    precision = precision_score(y_true, y_pred)
    # Calculate recall
    recall = recall_score(y_true, y_pred)
    # Calculate f1 score
    f1 = f1_score(y_true, y_pred)
    print("Accuracy, Precision, Recall, F1", accuracy, precision, recall, f1)



calc_scores()
