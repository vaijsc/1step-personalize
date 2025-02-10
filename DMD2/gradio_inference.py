import torch
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    LCMScheduler,
)
from huggingface_hub import hf_hub_download
import gradio as gr
import os

base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "tianweiy/DMD2"
ckpt_name = "dmd2_sdxl_1step_unet_fp16.bin"
ip_adapter_model = "h94/IP-Adapter"
output_dir = "output_dmd2_ipa"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


@torch.no_grad()
def pipeline_inference(
    pipe: StableDiffusionXLPipeline,
    prompt,
    negative_prompt,
    guidance_scale,
    ip_adapter_image,
    ip_adapter_scale,
    generator,
):
    pipe.set_ip_adapter_scale(ip_adapter_scale)
    img = pipe(
        prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        ip_adapter_image=ip_adapter_image,
        generator=generator,
        timesteps=[399],
        num_inference_steps=1,
    ).images[0]
    return img


def save_image(image, prompt):
    image.save(f"output_dmd2_ipa/{prompt}.png")
    gr.Info(f"Image saved successfully to output_dmd2_ipa/{prompt}.png!")


def main():
    # Load model.
    unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to(
        "cuda", torch.float16
    )
    unet.load_state_dict(
        torch.load(hf_hub_download(repo_name, ckpt_name), map_location="cuda")
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_id, unet=unet, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter(
        ip_adapter_model,
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl.bin",
    )

    generator = torch.Generator(device="cuda")

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # DMD2 Text-to-Image Generation
            Generate high-quality images from text descriptions in a flash.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                ip_adapter_image = gr.Image(
                    label="IP Adapter Image",
                    height=512,
                    width=512,
                    sources=["upload", "clipboard"],
                )

                prompt = gr.Textbox(
                    label="Prompt", placeholder="Enter your prompt here...", lines=1
                )
                ip_adapter_scale = gr.Slider(
                    label="IP Adapter Scale",
                    value=0.4,
                    step=0.1,
                    minimum=0.0,
                    maximum=1.0,
                )
                randomize = gr.Checkbox(label="Randomize seed", value=False)
                with gr.Accordion():
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="Enter your negative prompt here...",
                        lines=1,
                        value="Blurry, noisy, pixelated, distorted, low resolution, low quality, artifacts, deformed, ugly, asymmetry, unnatural lighting, text, watermark, oversaturated, low contrast, lack of detail.",
                    )

                    # with gr.Column():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        value=0,
                        step=0.1,
                        minimum=0.0,
                        maximum=10.0,
                    )

                    seed = gr.Number(label="Seed", value=42, precision=0)

                with gr.Row():
                    generate_btn = gr.Button("Generate Image")
                    save_image_btn = gr.Button("Save Image")

            output_image = gr.Image(label="Generated Image", height=768, width=768)

            def inference_wrapper(
                prompt_text,
                negative_prompt,
                guidance_scale,
                ip_adapter_image,
                ip_adapter_scale,
                seed_val,
                randomize_val,
            ):
                if randomize_val:
                    generator.manual_seed(int(torch.randint(0, 1000000, (1,)).item()))
                else:
                    generator.manual_seed(int(seed_val))
                # return inference(
                #     pipe, encode_prompt, prompt_text, generator, device, dtype
                # )
                return pipeline_inference(
                    pipe,
                    prompt_text,
                    negative_prompt,
                    guidance_scale,
                    ip_adapter_image,
                    ip_adapter_scale,
                    generator,
                )

            generate_btn.click(
                fn=inference_wrapper,
                inputs=[
                    prompt,
                    negative_prompt,
                    guidance_scale,
                    ip_adapter_image,
                    ip_adapter_scale,
                    seed,
                    randomize,
                ],
                outputs=output_image,
            )
            save_image_btn.click(save_image, inputs=[output_image, prompt])

    demo.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
