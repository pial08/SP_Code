

# DevSecLLM: Repairing Obfuscated Code Vulnerabilities using Reinforcement Learning from Human Feedback 

The emergence of transformer-based Large Language Models (LLMs) has drastically reshaped the domain of code development. The proliferation of automated tools that expedite the creation of code snippets has streamlined developers' workflows. However, this enhanced accessibility introduces escalated security challenges within the field of software development. In open-source solutions and code-generation systems, functionality is often given precedence over security, leading to the potential manifestation of security vulnerabilities. The situation is further exacerbated when adversaries harness the power of LLMs, capable of generating obfuscated codes that conceal embedded vulnerabilities. These tactics unintentionally grant malicious codes unrestricted access to critical system functionalities. The predicament is worsened due to a widespread deficiency in security literacy among developers. This research introduces an innovative method that harnesses the capabilities of Large Language Models, integrated with human feedback, alongside a unique instruct-based dataset, for detecting code vulnerabilities within obfuscated code. This approach not only identifies vulnerabilities but also facilitates code repair for their remediation. Moreover, it offers developers comprehensive descriptions to enrich their understanding of the identified security deficiencies in software development. As a practical application, we demonstrate the efficacy of our system in identifying, repairing, and describing 5 zero-day and 30 N-day vulnerabilities from the source code of IoT operating systems.





#### Requirements
- Python 	3.7
- Pytorch 	1.9 
- Transformer 	4.4
- torchmetrics 0.11.4
- tree-sitter 0.20.1
- sctokenizer 0.0.8

Moreover the above libraries can be installed by the commands from *requirements.txt* file. It is assumed that the installation will be done in a Linux system with a GPU. If GPU does not exist please remove the first command from the *requirements.txt*  file and replace it with 

`conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 -c pytorch` for OSX

or 


`conda install pytorch==1.9.0 torchvision==0.10.1 torchaudio==0.9.1 cpuonly -c pytorch` for Linux and Windows with no GPU.

Instructions to install libraries using *requirements.txt* file.

```shell
cd code 
pip install -r requirements.txt
```


### Usage
The repository is partially based on [CodeXGLUE](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection).


### Training and Evaluation
The following command should be used for training, testing and evaluation. Please set the ```--output_dir``` to the address where the model will be saved. We have also compiled a shell file with the same command for ease of use for the practitioners. Please put the location/address of train, evaluation and test file directory for the parameters
```--train_data_file```, ```--eval_data_file``` and ```--test_data_file```. 


Please run the following commands:

```shell


./run.sh

or,

torchrun --nproc_per_node=4 --master_port=1234 train.py \
    --model_name_or_path Salesforce/codegen2-3_7B \
    --data_path ./andrew/instruct-repair-bigvul-train.json \
    --output_dir results_codegen2_3_7_bigvul_repair_train \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --gradient_checkpointing false \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --fsdp "full_shard offload auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'CodeGenBlock' \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fp16 True

```




### Datasets
- Please download our [InstructVul](https://drive.google.com/drive/folders/1A7vfAkImoX8yvZCeTtqZ_45aqF-r214K?usp=sharing) dataset VulF directory.


- Our N-day and zero-day samples are also available in the previous link under *Testing* directory.
- After downloading VulF dataset, please put it under the directory *data*.

### Reproducibility
In order to use our pre-trained model, please download our model from [here](https://drive.google.com/drive/folders/1A7vfAkImoX8yvZCeTtqZ_45aqF-r214K?usp=sharing) under the Saved Model directory. After downloading, please set the value of the parameter `--model_checkpoint` to local directory you saved the pre-trained model.

## Cite  
Please cite the paper whenever our ReGVD is used to produce published results or incorporated into other software:

 



		

## License
As a free open-source implementation, our repository is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. All other warranties including, but not limited to, merchantability and fitness for purpose, whether express, implied, or arising by operation of law, course of dealing, or trade usage are hereby disclaimed. I believe that the programs compute what I claim they compute, but I do not guarantee this. The programs may be poorly and inconsistently documented and may contain undocumented components, features or modifications. I make no guarantee that these programs will be suitable for any application.

