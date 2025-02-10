from diffusers import UNet2DConditionModel, LCMScheduler, StableDiffusionPipeline
from diffusers.utils import load_image
import torch
from PIL import Image
import gradio as gr
from huggingface_hub import hf_hub_download

from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0

from diffusers.image_processor import IPAdapterMaskProcessor

layout_range = (0.0, 1.0)

mask = torch.zeros((512, 512))
mask[
    int(512 * layout_range[0]) : int(512 * layout_range[1]),
    int(512 * layout_range[0]) : int(512 * layout_range[1]),
] = 1
processor = IPAdapterMaskProcessor()
masks = processor.preprocess([mask], height=512, width=512)


def set_multiscale_ipa(pipe, scales=[0.0, 0.25, 0.5, 1.0, 1.0]):
    """Sets IP adapter scale gradually decreasing from lower to higher layers."""

    # Get all attention processor keys
    attn_procs = pipe.unet.attn_processors
    down_blocks = [k for k in attn_procs.keys() if k.startswith("down_blocks")]
    mid_blocks = [k for k in attn_procs.keys() if k.startswith("mid_block")]
    up_blocks = [k for k in attn_procs.keys() if k.startswith("up_blocks")]

    # Calculate scales for each level
    num_levels = (
        len(down_blocks) // 4 + 1
    )  # Each block has 2 attention modules and the last layer of down blocks does not have cross attention

    assert len(scales) > num_levels

    # Set scales for down blocks (high to low)
    for i in range(num_levels):
        # Get both attention modules for this level
        attn_keys = [
            k
            for k in down_blocks
            if f"down_blocks.{i}." in k
            and isinstance(attn_procs[k], IPAdapterAttnProcessor2_0)
        ]
        for key in attn_keys:
            attn_procs[key].scale = [scales[i]]

    # Set scale for mid block
    for block in mid_blocks:
        if isinstance(attn_procs[block], IPAdapterAttnProcessor2_0):
            attn_procs[block].scale = [scales[-1]]

    # Set scales for up blocks (low to high)
    num_up_levels = len(up_blocks) // 6 + 1
    for i in range(num_up_levels):
        # Get both attention modules for this level
        attn_keys = [
            k
            for k in up_blocks
            if f"up_blocks.{i}." in k
            and isinstance(attn_procs[k], IPAdapterAttnProcessor2_0)
        ]
        level = num_up_levels - 1 - i  # Reverse the levels
        for key in attn_keys:
            attn_procs[key].scale = [scales[level]]

    print("Loaded IP Adapter with multiscales:", scales)


pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")
pipeline.load_ip_adapter(
    "h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin"
)

# set_multiscale_ipa(pipeline)

generator = torch.Generator(device="cuda")


@torch.no_grad()
def infer(
    prompt,
    negative_prompt,
    guidance_scale,
    ip_adapter_image,
    ip_adapter_scale,
    seed,
    randomize_seed,
    num_inference_steps,
):
    pipeline.set_ip_adapter_scale(ip_adapter_scale)

    if randomize_seed:
        seed = torch.randint(0, 2147483647, (1,)).item()
        generator.manual_seed(seed)
    else:
        generator.manual_seed(seed)
    print(
        f"seed: {seed}, ip_adapter_scale: {ip_adapter_scale}, guidance_scale: {guidance_scale}, prompt: {prompt}"
    )
    images = pipeline(
        prompt=prompt,
        ip_adapter_image=ip_adapter_image,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        cross_attention_kwargs={"ip_adapter_masks": masks},
    ).images

    return images[0]


def save_image(image, prompt):
    image.save(f"output_sd_ipa/{prompt}.png")
    gr.Info(f"Image saved successfully to output_sd_ipa/{prompt}.png!")


with gr.Blocks() as demo:
    gr.Markdown("Stable Diffusion inference with IP-Adapter")
    with gr.Row():
        with gr.Column():
            ip_adapter_image = gr.Image(
                label="IP-Adapter Image",
                sources=["upload", "clipboard"],
                height=768,
                width=768,
            )
            prompt_input = gr.Textbox(
                value="A high-resolution photograph of a waterfall in autumn; muted tone",
                label="Prompt",
            )
            ip_adapter_scale = gr.Slider(
                minimum=0,
                maximum=1,
                label="IP-Adapter Scale",
                value=0.4,
                step=0.1,
            )
            randomize_seed = gr.Checkbox(
                label="Randomly Sample a Random Seed", value=True
            )
            with gr.Accordion("Generation Options", open=False):
                seed_input = gr.Slider(
                    value=42, minimum=0, maximum=2147483647, label="Random Seed"
                )
                num_inference_steps = gr.Slider(
                    value=25,
                    minimum=1,
                    maximum=100,
                    step=1,
                    label="Number of Inference Steps",
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="Enter your negative prompt here...",
                    lines=1,
                    value="Blurry, noisy, pixelated, distorted, low resolution, low quality, artifacts, deformed, ugly, asymmetry, unnatural lighting, text, watermark, oversaturated, low contrast, lack of detail.",
                )

                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    value=4.5,
                    step=0.1,
                )

        with gr.Column():
            im = gr.Image(type="pil", height=768, width=768)
            with gr.Row():
                new_image_button = gr.Button(value="Generate", variant="primary")
                save_image_button = gr.Button(value="Save")

                new_image_button.click(
                    infer,
                    inputs=[
                        prompt_input,
                        negative_prompt,
                        guidance_scale,
                        ip_adapter_image,
                        ip_adapter_scale,
                        seed_input,
                        randomize_seed,
                        num_inference_steps,
                    ],
                    outputs=[im],
                )
                save_image_button.click(
                    save_image, inputs=[im, prompt_input], outputs=None
                )

demo.launch(server_name="0.0.0.0", server_port=7860)
