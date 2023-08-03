#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer


import gc
import pandas as pd
from torchmetrics import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from pprint import pprint
from evaluate import load
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, AutoModelForSeq2SeqLM, T5ForConditionalGeneration


IGNORE_INDEX = -100
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


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


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


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")

        #print("Printing an element of sources ...", sources[0])
        #print("Printing a target ...", targets[0])

        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
            print("Label Found ...")
        else:
            labels = None
        outputs = model(**inputs)
        print(labels, outputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Nafis: Changed line 190 and 191 for 8 bit training for the llama 33B model
    
    # For Causal Models 
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )
    
    # For T5 Type models
    #model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    print("Printing model ...", model)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)




    trainer.train()
    trainer.save_state()

    #evaluate(model, tokenizer)

    trainer.save_model(output_dir=training_args.output_dir)
    print("--------------------- Model Saving Complete ----------------")


def read_data(data_path, model, tokenizer):
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


def evaluate(model, tokenizer):

    file_path = "data/andrew/instruct-repair-test.json"
    #df = pd.read_csv(file_path)

    total = counter = 300


    blue_score = 0
    gblue = 0
    rouge = 0
    false_counter = 0


    sources, targets = read_data(file_path, model, tokenizer)

    print(sources[0])
    print(targets[0])

    print(len(sources) == len(targets))



    #print("Input ...", tokenizer.batch_decode(data_dict["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
    #print("Label ...", tokenizer.batch_decode(data_dict["labels"], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

    #return

    generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)

    for i in range(len(sources)):



        response = generator(sources[i], max_length=512, do_sample=False)[0]['generated_text']

        #output = bytes(output, "utf-8").decode("unicode_escape")

        original_input = sources[i] #bytes(sources[i], "utf-8").decode("unicode_escape")
        model_output = response #bytes(response, "utf-8").decode("unicode_escape")
        original_output = targets[i] #bytes(targets[i], "utf-8").decode("unicode_escape")
        """
        print("----------------------------------------------------------")
        print("Original Input \n", original_input)
        print("Output from model\n", model_output)
        print("Original Output\n", original_output)
        print("----------------------------------------------------------")
        """



        """
        bleu = evaluate.load("bleu")
        results = bleu.compute(predictions=output, references=original_output.split("/~/")[0]) #.split("/~/")[0]
        blue_score += results["bleu"]
        print("BLUE Score:", results["bleu"])

        """
        rouge = ROUGEScore()
        results = rouge(model_output, original_output)
        rouge += results["rouge1_fmeasure"]
        print("Rouge Score ", results["rouge1_fmeasure"])


        google_bleu = evaluate.load("google_bleu")
        result = google_bleu.compute(predictions=[model_output], references=[[original_output]])
        gblue += result["google_bleu"]
        print("Google Blue Score", result["google_bleu"])

        counter -= 1
        if counter == 0:
            break

    #print("Avg. Blue = ", blue_score / counter)
    print
    print("Avg Rouge = ", rouge / total)
    print("Total Blue ...", gblue)
    print("Avg GBLUE ", gblue / total)

    print("False Counter ...", false_counter)




if __name__ == "__main__":
    train()

"""
if self.is_fsdp_enabled:
2853                 os.makedirs(output_dir, exist_ok=True)
2854                 self.accelerator.state.fsdp_plugin.save_model(self.accelerator, self.model, output_dir)
2855             else:
2856                 state_dict = self.model.state_dict()

"""