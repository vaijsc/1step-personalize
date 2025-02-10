export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export OUTPUT_DIR="dreambooth-output-sb-no-pertubation"
export TRAIN_DATA_DIR="/root/research/1-step-personalize/Data/dreambench/can"

accelerate launch --num_processes 1 --mixed_precision "fp16" \
    train_dreambooth_sb.py \
  --pretrained_model_name_or_path $MODEL_NAME \
  --instance_data_dir $TRAIN_DATA_DIR \
  --instance_prompt "a photo of sks can" \
  --output_dir $OUTPUT_DIR \
  --seed 42 \
  --resolution 512 \
  --train_batch_size 16 \
  --num_train_epochs 30 \
  --max_train_steps 1000 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-6 \
  --scale_lr \
  --lr_scheduler "constant" --lr_warmup_steps 0 \
  --lr_num_cycles 1 \
  --dataloader_num_workers 8 \
  --mixed_precision "fp16" \
  --validation_prompt "a photo of sks can in Time Square" \
  --num_validation_images 4 \
  --validation_steps 100 \
  --checkpointing_steps 100
