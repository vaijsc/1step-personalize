import os
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, LCMScheduler
from PIL import Image
from transformers import CLIPVisionModelWithProjection
from transformers import CLIPImageProcessor
import gradio as gr
from gradio_image_prompter import ImagePrompter
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
log_id = "test"
ms_ckpt = f"./output/{log_id}/pretrained/ms_adapter.bin"
result_path = f"./output_gradio_os"
repo_name = "tianweiy/DMD2"
ckpt_name = "dmd2_sdxl_1step_unet_fp16.bin"

if not os.path.exists(result_path):
    os.makedirs(result_path)

image_processor = CLIPImageProcessor()

# load SDXL pipeline
unet = UNet2DConditionModel.from_config(base_model_path, subfolder="unet").to(
    "cuda", torch.float16
)
unet.load_state_dict(
    torch.load(
        hf_hub_download(repo_name, ckpt_name), map_location="cuda", weights_only=True
    )
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    unet=unet,
    torch_dtype=torch.float16,
    add_watermarker=False,
    variant="fp16",
)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
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


def gradio_infer(
    input_image,
    text_prompt,
    subject,
    scale,
    seed,
):
    img = [input_image]

    bbox = [[[0.25, 0.25, 0.75, 0.75]]]

    phrase_idxes = [get_phrases_idx(pipe.tokenizer, subject, text_prompt)]
    eot_idxes = [[get_eot_idx(pipe.tokenizer, text_prompt)] * len(subject)]

    return ms_model.generate(
        pipe=pipe,
        pil_images=[img],
        num_samples=1,
        num_inference_steps=1,
        guidance_scale=1 + 1e-4,
        seed=seed,
        prompt=[text_prompt],
        scale=scale,
        image_encoder=image_encoder,
        image_processor=image_processor,
        boxes=bbox,
        image_proj_type=image_proj_type,
        image_encoder_type=image_encoder_type,
        phrases=[[subject]],
        drop_grounding_tokens=[0],
        phrase_idxes=phrase_idxes,
        eot_idxes=eot_idxes,
        height=1024,
        width=1024,
        timesteps=[399],
        start_step=1,
        do_classifier_free_guidance=True,
    )[0]


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # MS-Diffusion Text-to-Image Generation
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            ref_image = gr.Image(
                label="Reference Image",
                height=768,
                width=768,
                type="pil",
            )

            prompt = gr.Textbox(
                label="Prompt", placeholder="Enter your prompt here...", lines=1
            )
            subject = gr.Textbox(
                label="Subject", placeholder="Enter your subject here...", lines=1
            )
            randomize = gr.Checkbox(label="Randomize seed", value=False)
            strength = gr.Slider(
                label="ID Strength",
                value=1,
                step=0.1,
                minimum=0.0,
                maximum=2.0,
            )

            seed = gr.Number(label="Seed", value=42, precision=0)

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Generated Image",
                height=768,
                width=768,
                show_download_button=True,
            )
            with gr.Row():
                generate_btn = gr.Button("Generate Image")
                save_btn = gr.Button("Save Image")

        def inference_wrapper(
            ref_image,
            prompt_text,
            subject,
            strength,
            seed,
            randomize_val,
        ):
            if randomize_val:
                seed = int(torch.randint(0, 1000000, (1,)).item())

            return gradio_infer(
                ref_image,
                prompt_text,
                subject,
                strength,
                seed,
            )

        def save_image(image, prompt):
            image = Image.fromarray(image)
            image.save(os.path.join(result_path, f"{prompt}.jpg"))
            gr.Info("Image saved to " + os.path.join(result_path, f"{prompt}.jpg"))

        generate_btn.click(
            fn=inference_wrapper,
            inputs=[
                ref_image,
                prompt,
                subject,
                strength,
                seed,
                randomize,
            ],
            outputs=output_image,
        )

        save_btn.click(
            fn=save_image,
            inputs=[output_image, prompt],
        )

demo.launch(server_name="0.0.0.0")
