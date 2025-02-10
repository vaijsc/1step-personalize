import gradio as gr
import torch
import os
from torchvision.transforms.functional import to_pil_image
from diffusers import (
    UNet2DConditionModel,
    DDPMScheduler,
)
from sb_pipeline import SwiftBrushPipeline

BASE_MODEL = "stabilityai/sd-turbo"
# UNET_MODEL = "ckpts/sbv2_fid81"
UNET_MODEL = "swiftbrush-output-dreambooth_can_1/checkpoint-41000"
# IP_ADAPTER_MODEL = "/root/research/1-step-personalize/Experiments/pretrained"


@torch.no_grad()
def pipeline_inference(
    pipe,
    prompt,
    negative_prompt,
    guidance_scale,
    # ip_adapter_image,
    # ip_adapter_scale,
    generator,
):
    # pipe.set_ip_adapter_scale(ip_adapter_scale)
    return pipe(
        prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        # ip_adapter_image=ip_adapter_image,
        generator=generator,
        num_inference_steps=1,
    ).images[0]


def save_image(image, prompt):
    os.makedirs("sb_ipa", exist_ok=True)
    image.save(f"sb_ipa/{prompt}.png")
    gr.Info(f"Image saved to 'sb_ipa/{prompt}.png'")


def main():
    device, dtype = "cuda", torch.float16

    scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(
        UNET_MODEL, subfolder="unet_ema", torch_dtype=dtype
    )

    pipe = SwiftBrushPipeline.from_pretrained(
        BASE_MODEL,
        scheduler=scheduler,
        unet=unet,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)
    # pipe.load_ip_adapter(
    #     IP_ADAPTER_MODEL,
    #     subfolder="",
    #     local_files_only=True,
    #     weight_name="ip_adapter_checkpoint-10000.bin",
    # )

    generator = torch.Generator(device=device)

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # SwiftBrushv2 Text-to-Image Generation
            Generate high-quality images from text descriptions in a flash.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                # ip_adapter_image = gr.Image(
                #     label="IP Adapter Image",
                #     height=512,
                #     width=512,
                #     sources=["upload", "clipboard"],
                # )

                prompt = gr.Textbox(
                    label="Prompt", placeholder="Enter your prompt here...", lines=1
                )
                # ip_adapter_scale = gr.Slider(
                #     label="IP Adapter Scale",
                #     value=0.4,
                #     minimum=0.0,
                #     maximum=1.0,
                #     step=0.1,
                # )

                randomize = gr.Checkbox(label="Randomize seed", value=False)

                with gr.Accordion("Advanced Settings"):
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="Enter your negative prompt here...",
                        lines=1,
                        value="Blurry, noisy, pixelated, distorted, low resolution, low quality, artifacts, deformed, ugly, asymmetry, unnatural lighting, text, watermark, oversaturated, low contrast, lack of detail.",
                    )

                    # with gr.Column():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        value=0.0,
                        minimum=0.0,
                        maximum=10.0,
                        step=0.1,
                    )

                    seed = gr.Number(label="Seed", value=42, precision=0)

            with gr.Column(scale=1):
                output_image = gr.Image(label="Generated Image", type="pil")
                with gr.Row():
                    generate_btn = gr.Button("Generate Image", variant="primary")
                    save_btn = gr.Button("Save Image")

            def inference_wrapper(
                prompt_text,
                negative_prompt,
                guidance_scale,
                # ip_adapter_image,
                # ip_adapter_scale,
                seed_val,
                randomize_val,
            ):
                if randomize_val:
                    generator.manual_seed(int(torch.randint(0, 1000000, (1,)).item()))
                else:
                    generator.manual_seed(int(seed_val))

                return pipeline_inference(
                    pipe,
                    prompt_text,
                    negative_prompt,
                    guidance_scale,
                    # ip_adapter_image,
                    # ip_adapter_scale,
                    generator,
                )

            generate_btn.click(
                fn=inference_wrapper,
                inputs=[
                    prompt,
                    negative_prompt,
                    guidance_scale,
                    # ip_adapter_image,
                    # ip_adapter_scale,
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


if __name__ == "__main__":
    main()
