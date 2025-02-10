import torch
from diffusers import (
    DDPMScheduler,
    UNet2DConditionModel,
)
from huggingface_hub import hf_hub_download
from PIL import Image
import json
import os

from sb_pipeline import SwiftBrushPipeline
from diffusers.image_processor import IPAdapterMaskProcessor


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


mask = torch.zeros((512, 512))
mask[int(512 * 0.25) : int(512 * 0.75), int(512 * 0.25) : int(512 * 0.75)] = 1
processor = IPAdapterMaskProcessor()
masks = processor.preprocess([mask], height=512, width=512)

train_data_dir = "/root/research/1-step-personalize/Data/dreambench"
base_model_path = "runwayml/stable-diffusion-v1-5"
base_unet_path = "/root/research/os-personalize/SB/ckpts/sbv2_sd1.5/0.7/unet"
device = "cuda"
seed = 42

# load SwiftBrushV2 pipeline

unet = UNet2DConditionModel.from_pretrained(
    base_unet_path,
    subfolder="",
).to(device, torch.float16)
scheduler = DDPMScheduler.from_pretrained(base_model_path, subfolder="scheduler")


pipe = SwiftBrushPipeline.from_pretrained(
    base_model_path,
    unet=unet,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    safety_checker=None,
).to(device)

pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="models",
    weight_name="ip-adapter-plus_sd15.bin",
)
generator = torch.Generator(device=device).manual_seed(seed)

if __name__ == "__main__":
    with open(os.path.join(train_data_dir, "test_dataset_ref.json"), "r") as f:
        data = json.load(f)

    for scale in [1.0, 1.2, 1.4]:
        print(f"Generating images with scale {scale}")
        pipe.set_ip_adapter_scale(scale)
        for object in data:
            ref_path = object["ref_path"]

            if ref_path not in ["can", "dog2", "colorful_sneaker"]:
                continue

            print(f"Object: {ref_path}")

            prompts = object["prompts"]
            negative_prompt = "Blurry, noisy, pixelated, distorted, low resolution, low quality, artifacts, deformed, ugly, asymmetry, unnatural lighting, text, watermark, oversaturated, low contrast, lack of detail."

            os.makedirs(f"results_sb/{scale}/{ref_path}", exist_ok=True)

            ip_adapter_image = Image.open(
                os.path.join(train_data_dir, ref_path, "00.jpg")
            )

            images = []
            for prompt in prompts:
                image = pipe(
                    prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=0,
                    ip_adapter_image=ip_adapter_image,
                    generator=generator,
                    # timesteps=[399],
                    num_inference_steps=1,
                    cross_attention_kwargs={"ip_adapter_masks": masks},
                ).images[0]
                images.append(image)

            grid = image_grid(images, 5, 5)
            grid.save(f"results_sb/{scale}/{ref_path}/grid.png")

            for image, prompt in zip(images, prompts):
                image.save(f"results_sb/{scale}/{ref_path}/{prompt}.png")
