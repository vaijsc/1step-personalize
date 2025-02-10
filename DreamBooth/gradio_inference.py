import gradio as gr
import torch
from torchvision.transforms.functional import to_pil_image
from diffusers import UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline

BASE_MODEL = "stabilityai/stable-diffusion-2-1-base"
# UNET_MODEL = "ckpts/sbv2_fid81"
# UNET_MODEL = (
#     "/root/research/1-step-personalize/DreamBooth/dreambooth-output-sd/checkpoint-1000"
# )
UNET_MODEL = "/root/research/1-step-personalize/Experiments/outputs/dreambooth-output-sd/checkpoint-100"


@torch.no_grad()
def pipeline_inference(
    pipe,
    prompt,
    negative_prompt,
    guidance_scale,
    generator,
):
    return pipe(
        prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        generator=generator,
        num_inference_steps=25,
    ).images[0]


def save_image(image, prompt):
    image.save(f"sd_dreambooth/{prompt}.png")
    gr.Info(f"Image saved to 'sd_dreambooth/{prompt}.png'")


def main():
    device, dtype = "cuda", torch.float16

    unet = UNet2DConditionModel.from_pretrained(
        UNET_MODEL, subfolder="unet", torch_dtype=dtype
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        unet=unet,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)

    generator = torch.Generator(device=device)

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # Stable Diffusion Text-to-Image Generation
            """
        )
        with gr.Row():
            with gr.Column(scale=1):

                prompt = gr.Textbox(
                    label="Prompt", placeholder="Enter your prompt here...", lines=1
                )

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
                        value=4.5,
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
                    generator,
                )

            generate_btn.click(
                fn=inference_wrapper,
                inputs=[
                    prompt,
                    negative_prompt,
                    guidance_scale,
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
