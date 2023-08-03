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


# importing datetime module
from datetime import datetime
 


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

#incorrect name, correct model
trained_model = "results_codegen2_desc_train"

# device_map="auto"
# Removed device_map = "auto" for santacoder model
# ignore_mismatched_sizes=True
model = AutoModelForCausalLM.from_pretrained(trained_model, trust_remote_code=True, device_map = "auto")

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

    file_path = "andrew/instruct-desc-test.json"
    #df = pd.read_csv(file_path)

    total = counter = 5


    blue_score = 0
    gblue = 0
    rouge_score = 0
    bert_score = 0
    false_counter = 0


    sources, targets = read_data(file_path)

    print(sources[0])
    print(targets[0])

    print(len(sources) == len(targets))



    
    myobj = datetime.now()
    
    
    

    #print("Input ...", tokenizer.batch_decode(data_dict["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
    #print("Label ...", tokenizer.batch_decode(data_dict["labels"], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

    #return

    # Loading the Scores
    rouge = ROUGEScore()
    google_bleu = evaluate.load("google_bleu")
    bertscore = load("bertscore")


    this_temperature = float(input("Enter a temperature value: "))
    print("Temperature value is", this_temperature)

    

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
        """
        # Generate
        # Next 4 are good lines
        

        try:
            
            time_start =  myobj.minute
            print("Printing original input variable here...", sources[i])
            inputs = tokenizer(str(sources[i]), return_tensors="pt", max_length=512, truncation=True).to(device)
            generate_ids = model.generate(inputs.input_ids, max_new_tokens=512, num_beams=1, do_sample=True, early_stopping=True, pad_token_id=tokenizer.eos_token_id)
            response = model_output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            

        #original_output = row["func_after"].strip()
        
            

            # for the pipeline
            #response = generator(sources[i], max_length=512, num_beams=1,temperature = 0.0, do_sample=False)[0]['generated_text']

            time_end =  myobj.minute
            #output = bytes(output, "utf-8").decode("unicode_escape")

            original_input = sources[i] #bytes(sources[i], "utf-8").decode("unicode_escape")
            model_output = response#.split("### Response:")[1] #bytes(response, "utf-8").decode("unicode_escape")
            original_output = targets[i] #bytes(targets[i], "utf-8").decode("unicode_escape")
            
            print("----------------------------------------------------------")
            #print("Original Input \n", original_input)
            #print("Output from model\n", model_output)
            #print("Original Output\n", original_output)
            print("Time elapsed ", time_end - time_start)
            print("----------------------------------------------------------")
            



            """
            bleu = evaluate.load("bleu")
            results = bleu.compute(predictions=model_output, references=original_output) #.split("/~/")[0]
            blue_score += results["bleu"]
            print("BLUE Score:", results["bleu"])
            """
            
            
            results = rouge(model_output, original_output)
            rouge_score += results["rouge1_fmeasure"]
            print("Rouge Score ", results["rouge1_fmeasure"])

            
            result = google_bleu.compute(predictions=[model_output], references=[[original_output]])
            gblue += result["google_bleu"]
            print("Google BLEU Score", result["google_bleu"])

            bertscore = load("bertscore")
            results = bertscore.compute(predictions=[model_output], references=[[original_output]], lang="en")
            bert_score += results['f1'][0]
            print("Results BERT Score", results['f1'][0])

            counter -= 1
            if counter == 0:
                break
        except:
            pass

    #print("Avg. Blue = ", blue_score / counter)
    print("Total Rouge Score = ", rouge_score)
    print("Avg Rouge = ", rouge_score / total)
    print("Total Blue ...", gblue)
    print("Avg GBLUE ", gblue / total)
    print("Avg Bert Score ", bert_score / total)


    print("False Counter ...", false_counter)



calc_scores()
