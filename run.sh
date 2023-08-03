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


# ./andrew/instruct-repair-cvefixes-train.json \
# Salesforce/codegen2-7B
# Salesforce/codegen-350M-mono
#--bf16 True \
#--fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
# CodeGenModel
# Salesforce/codegen-6B-mono
# --gradient_checkpointing True \
# Salesforce/codet5-large
# -- fsdp_transformer_layer_cls_to_wrap 'T5Block'
# --fsdp_transformer_layer_cls_to_wrap 'CodeGenBlock'
# original LR --learning_rate 2e-5 \
# bigcode/gpt_bigcode-santacoder
# fsdp_transformer_layer_cls_to_wrap 'GPTBigCodeBlock'
# ./alpaca_data.json
# --fsdp_transformer_layer_cls_to_wrap GPTJBlock
## Inquire test train data

#  --fsdp "full_shard auto_wrap" \
# --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
# --fp16 --sharded_ddp "zero_dp_3"
# --tf32 True


# --fsdp "full_shard offload auto_wrap" \
# --fsdp_transformer_layer_cls_to_wrap 'CodeGenBlock' \


# --deepspeed "configs/dgx_config_arc.json"

# Wrap for T5s CodeT5pBlock

# Salesforce/codegen-6B-multi
# bigcode/santacoder
