import torch
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    LCMScheduler,
)
from huggingface_hub import hf_hub_download
import gradio as gr
import os
from PIL import Image

base_model_id = "runwayml/stable-diffusion-v1-5"
repo_name = "tianweiy/DMD2"
ckpt_name = "model/sdv1.5/laion6.25_sd_baseline_8node_guidance1.75_lr5e-7_seed10_dfake10_diffusion1000_gan1e-3_resume_fid8.35_checkpoint_model_041000/pytorch_model.bin"
ip_adapter_model = "h94/IP-Adapter"
output_dir = "output_dmd2_ipa_v1-5"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


@torch.no_grad()
def pipeline_inference(
    pipe: StableDiffusionPipeline,
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
        # timesteps=[399],
        num_inference_steps=1,
    ).images[0]
    img.save(f"{output_dir}/{prompt}.png")
    return img


def save_image(img, prompt):
    img = Image.fromarray(img)
    img.save(f"{output_dir}/{prompt}.png")
    gr.Info(f"Image saved to {output_dir}/{prompt}.png", duration=5)


def main():
    # Load model.
    unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet").to(
        "cuda", torch.float16
    )
    unet.load_state_dict(
        torch.load(hf_hub_download(repo_name, ckpt_name), map_location="cuda")
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        unet=unet,
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None,
    ).to("cuda")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # pipe.load_ip_adapter(
    #     ip_adapter_model,
    #     subfolder="models",
    #     weight_name="ip-adapter-plus_sd15.bin",
    # )

    # WEIGHT_PATH = "/root/research/1-step-personalize/Experiments/pretrained/dmd2-ip_adapter_checkpoint-10000.bin"
    # WEIGHT_PATH = "/root/research/1-step-personalize/Experiments/pretrained/finetuned_dmd2-ip_adapter_checkpoint-10000.bin"
    # WEIGHT_PATH = "/root/research/1-step-personalize/Experiments/distilled/dmd2-ip_adapter_checkpoint-4710.bin"

    # pipe.load_ip_adapter(
    #     "/root/research/1-step-personalize/Experiments/pretrained",
    #     subfolder="",
    #     weight_name=WEIGHT_PATH,
    # )
    # print(f"IP Adapter loaded from {WEIGHT_PATH}")

    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="models",
        weight_name="ip-adapter-plus_sd15.bin",
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
                randomize = gr.Checkbox(label="Randomize seed", value=False)
                ip_adapter_scale = gr.Slider(
                    label="IP Adapter Scale",
                    value=0.4,
                    step=0.1,
                    minimum=0.0,
                    maximum=1,
                )

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
                        value=0,
                        step=0.1,
                        minimum=0.0,
                        maximum=10.0,
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

            save_btn.click(
                fn=save_image,
                inputs=[output_image, prompt],
            )

    demo.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
