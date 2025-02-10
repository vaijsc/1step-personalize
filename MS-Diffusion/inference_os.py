import os
import torch
from diffusers import StableDiffusionXLPipeline, LCMScheduler
from PIL import Image
from transformers import CLIPVisionModelWithProjection
from transformers import CLIPImageProcessor
from huggingface_hub import hf_hub_download

from msdiffusion.models.projection import Resampler
from msdiffusion.models.model import MSAdapter
from msdiffusion.utils import get_phrase_idx, get_eot_idx


def get_phrases_idx(tokenizer, phrases, prompt):
    res = []
    phrase_cnt = {}
    for phrase in phrases:
        if phrase in phrase_cnt:
            cur_cnt = phrase_cnt[phrase]
            phrase_cnt[phrase] += 1
        else:
            cur_cnt = 0
            phrase_cnt[phrase] = 1
        res.append(get_phrase_idx(tokenizer, phrase, prompt, num=cur_cnt)[0])
    return res


base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
device = "cuda"
result_path = "./res_os"
log_id = "test"
load_type = "pretrained"
ms_ckpt = f"./output/{log_id}/{load_type}/ms_adapter.bin"
repo_name = "tianweiy/DMD2"
ckpt_name = "dmd2_sdxl_1step_unet_fp16.bin"

image_processor = CLIPImageProcessor()

# load SDXL pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    variant="fp16",
    torch_dtype=torch.float16,
    add_watermarker=False,
)
# pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to(device)

image_encoder_type = "clip"
image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(
    device, dtype=torch.float16
)
image_encoder_projection_dim = image_encoder.config.projection_dim
num_tokens = 16
image_proj_type = "resampler"
latent_init_mode = "grounding"
image_proj_model = Resampler(
    dim=1280,
    depth=4,
    dim_head=64,
    heads=20,
    num_queries=num_tokens,
    embedding_dim=image_encoder.config.hidden_size,
    output_dim=pipe.unet.config.cross_attention_dim,
    ff_mult=4,
    latent_init_mode=latent_init_mode,
    phrase_embeddings_dim=pipe.text_encoder.config.projection_dim,
).to(device, dtype=torch.float16)
ms_model = MSAdapter(
    pipe.unet, image_proj_model, ckpt_path=ms_ckpt, device=device, num_tokens=num_tokens
)
ms_model.to(device, dtype=torch.float16)

image0 = Image.open("/root/research/1-step-personalize/Data/dreambench/can/00.jpg")
# image0 = Image.open("./examples/example_cat.jpg")
input_images = [image0]
input_images = [x.convert("RGB").resize((512, 512)) for x in input_images]

# generation configs
num_samples = 5
prompt = "a can on the beach"
phrases = [["can"]]
print(prompt)
boxes = [[[0.25, 0.25, 0.75, 0.75]]]  # dog
# boxes = [[[0., 0.25, 0.4, 0.75], [0.6, 0.25, 1., 0.75]]]  # dog+cat
# boxes = [
#     [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
# ]  # used if you want no layout guidance
drop_grounding_tokens = [0]  # set to 1 if you want to drop the grounding tokens

# used to get the attention map, return zero if the phrase is not in the prompt
phrase_idxes = [get_phrases_idx(pipe.tokenizer, phrases[0], prompt)]
eot_idxes = [[get_eot_idx(pipe.tokenizer, prompt)] * len(phrases[0])]
print(phrase_idxes, eot_idxes)

images = ms_model.generate(
    pipe=pipe,
    pil_images=[input_images],
    num_samples=num_samples,
    seed=0,
    prompt=[prompt],
    scale=1,
    image_encoder=image_encoder,
    image_processor=image_processor,
    boxes=boxes,
    image_proj_type=image_proj_type,
    image_encoder_type=image_encoder_type,
    phrases=phrases,
    drop_grounding_tokens=drop_grounding_tokens,
    phrase_idxes=phrase_idxes,
    eot_idxes=eot_idxes,
    height=1024,
    width=1024,
    num_inference_steps=1,
    guidance_scale=0.0,
    timesteps=[399],
)

save_name = "dog"
save_path = os.path.join(result_path, log_id, load_type, save_name)
os.makedirs(save_path, exist_ok=True)
for i, image in enumerate(images):
    image.save(os.path.join(save_path, f"{prompt}_{i}.jpg"))
