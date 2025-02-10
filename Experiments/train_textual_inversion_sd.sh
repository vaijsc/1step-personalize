export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="textual-inversion-output_SD"
export TRAIN_DATA_DIR="/root/research/1-step-personalize/Data/dreambench/can"
export INITIALIZER_TOKEN="can"
export PLACEHOLDER_TOKEN="<can>"

accelerate launch --num_processes 1 --mixed_precision "fp16" \
    textual-inversion-multisteps.py \
  --save_steps 100 \
  --num_vectors 5 \
  --pretrained_model_name_or_path $MODEL_NAME \
  --train_data_dir $TRAIN_DATA_DIR \
  --initializer_token $INITIALIZER_TOKEN \
  --placeholder_token $PLACEHOLDER_TOKEN \
  --learnable_property "object" \
  --repeats 10 \
  --output_dir $OUTPUT_DIR \
  --seed 42 \
  --resolution 512 \
  --train_batch_size 16 \
  --num_train_epochs 30 \
  --max_train_steps 5000 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --scale_lr \
  --lr_scheduler "constant" --lr_warmup_steps 0 \
  --lr_num_cycles 1 \
  --dataloader_num_workers 8 \
  --mixed_precision "fp16" \
  --validation_prompt "a photo of a {} in Time Square" \
  --num_validation_images 4 \
  --validation_steps 100 \
  --checkpointing_steps 100 \
  --no_safe_serialization
