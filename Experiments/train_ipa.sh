accelerate launch --num_processes 2 --multi_gpu --mixed_precision "fp16" \
  tutorial_train.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --output_dir="dmd2-ip_adapter" \
  --data_json_file="unique_info_5k.json" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=24 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --save_steps=1000 \
  --num_train_epochs=100
