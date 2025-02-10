import torch
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    AutoencoderKL,
)
from PIL import Image
import json
import os
from diffusers.image_processor import IPAdapterMaskProcessor

mask = torch.zeros((512, 512))
mask[int(512 * 0.25) : int(512 * 0.75), int(512 * 0.25) : int(512 * 0.75)] = 1
processor = IPAdapterMaskProcessor()
masks = processor.preprocess([mask], height=512, width=512)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


train_data_dir = "/root/research/1-step-personalize/Data/dreambench"
base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "h94/IP-Adapter"
ip_ckpt = "ip-adapter-plus_sd15.bin"
device = "cuda"
seed = 42

# load SD pipeline
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    vae=vae,
    torch_dtype=torch.float16,
    feature_extractor=None,
    safety_checker=None,
).to(device)

pipe.scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

pipe.load_ip_adapter(
    image_encoder_path,
    subfolder="models",
    weight_name=ip_ckpt,
)
generator = torch.Generator(device="cuda").manual_seed(seed)

if __name__ == "__main__":
    with open(
        "/root/research/1-step-personalize/Data/dreambench/test_dataset_ref.json"
    ) as f:
        data = json.load(f)

    for scale in [1.0]:
        print(f"Generating images with scale {scale}")
        pipe.set_ip_adapter_scale(scale)
        for object in data:

            ref_path = object["ref_path"]

            if ref_path not in ["can", "dog2", "colorful_sneaker"]:
                continue

            print(f"Object: {ref_path}")

            prompts = object["prompts"]
            negative_prompt = "Blurry, noisy, pixelated, distorted, low resolution, low quality, artifacts, deformed, ugly, asymmetry, unnatural lighting, text, watermark, oversaturated, low contrast, lack of detail."

            os.makedirs(f"results_sd_1/{scale}/{ref_path}", exist_ok=True)

            ip_adapter_image = Image.open(
                os.path.join(train_data_dir, ref_path, "00.jpg")
            )

            images = []
            for prompt in prompts:
                image = pipe(
                    prompt,
                    ip_adapter_image=ip_adapter_image,
                    negative_prompt=negative_prompt,
                    guidance_scale=4.5,
                    num_inference_steps=25,
                    generator=generator,
                    cross_attention_kwargs={"ip_adapter_masks": masks},
                ).images[0]
                images.append(image)

            grid = image_grid(images, 5, 5)
            grid.save(f"results_sd_1/{scale}/{ref_path}/grid.png")

            for image, prompt in zip(images, prompts):
                image.save(f"results_sd_1/{scale}/{ref_path}/{prompt}.png")
