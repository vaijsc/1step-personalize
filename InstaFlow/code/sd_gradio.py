from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils import load_image
import torch
from PIL import Image
import gradio as gr

pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")
pipeline.load_ip_adapter(
    "h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin"
)
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
        safety_checker=None,
        generator=generator,
    ).images

    return images[0]


def save_image(image, prompt):
    image.save(f"output_sd_ipa/{prompt}.png")
    gr.Info(f"Image saved successfully to output_sd_ipa/{prompt}.png!")


with gr.Blocks() as demo:
    gr.Markdown("Stable Diffusion inference with IP-Adapter")
    with gr.Row():
        with gr.Column(scale=0.4):
            im = gr.Image(type="pil")
            with gr.Row():
                new_image_button = gr.Button(value="Generate", variant="primary")
                save_image_button = gr.Button(value="Save")

        with gr.Column(scale=0.4):
            ip_adapter_image = gr.Image(
                label="IP-Adapter Image", sources=["upload", "clipboard"]
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
            save_image_button.click(save_image, inputs=[im, prompt_input], outputs=None)


demo.launch(server_name="0.0.0.0", server_port=7860)
