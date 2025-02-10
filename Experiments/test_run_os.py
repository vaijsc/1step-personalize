import torch
from diffusers import (
    StableDiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)
from huggingface_hub import hf_hub_download
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
repo_name = "tianweiy/DMD2"
ckpt_name = "model/sdv1.5/laion6.25_sd_baseline_8node_guidance1.75_lr5e-7_seed10_dfake10_diffusion1000_gan1e-3_resume_fid8.35_checkpoint_model_041000/pytorch_model.bin"
device = "cuda"
seed = 42

# load DMD2 pipeline

unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet").to(
    "cuda", torch.float16
)
unet.load_state_dict(
    torch.load(hf_hub_download(repo_name, ckpt_name), map_location="cuda")
)
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    unet=unet,
    torch_dtype=torch.float16,
    variant="fp16",
    safety_checker=None,
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="models",
    weight_name="ip-adapter-plus_sd15.bin",
)
generator = torch.Generator(device="cuda").manual_seed(seed)

if __name__ == "__main__":
    with open(os.path.join(train_data_dir, "test_dataset_ref.json"), "r") as f:
        data = json.load(f)

    for scale in [1.0, 1.2, 1.4]:
        print(f"Generating images with scale {scale}")
        pipe.set_ip_adapter_scale(scale)
        for object in data:

            ref_path = object["ref_path"]

            if ref_path not in ["dog2", "can", "colorful_sneaker"]:
                continue

            print(f"Object: {ref_path}")

            prompts = object["prompts"]
            negative_prompt = "Blurry, noisy, pixelated, distorted, low resolution, low quality, artifacts, deformed, ugly, asymmetry, unnatural lighting, text, watermark, oversaturated, low contrast, lack of detail."

            os.makedirs(f"results_os/{scale}/{ref_path}", exist_ok=True)

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
            grid.save(f"results_os/{scale}/{ref_path}/grid.png")

            for image, prompt in zip(images, prompts):
                image.save(f"results_os/{scale}/{ref_path}/{prompt}.png")
