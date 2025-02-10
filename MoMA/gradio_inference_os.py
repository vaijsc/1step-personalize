import gradio as gr
from gradio_image_prompter import ImagePrompter
from PIL import Image, ImageDraw
import numpy as np

import os
import torch
from pytorch_lightning import seed_everything
import warnings

from model_lib.modules import DMD2_MoMA_main_modal
from model_lib.utils import parse_args

warnings.filterwarnings("ignore")

args = parse_args()
args.device = torch.device("cuda", 0)


# if you have 22 Gb GPU memory:
args.load_8bit, args.load_4bit = False, False

# if you have 18 Gb GPU memory:
# args.load_8bit, args.load_4bit = True, False

# if you have 14 Gb GPU memory:
# args.load_8bit, args.load_4bit = False, True


# load MoMA from HuggingFace. Auto download
moMA_main_modal = DMD2_MoMA_main_modal(args).to(args.device, dtype=torch.bfloat16)

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)


def inference(prompts, text_prompt, subject, strength, seed, return_mask=False):
    seed_everything(seed)

    image = Image.fromarray(prompts["image"])
    image.save("flagged/image.jpg")

    x1, y1, _, x2, y2, _ = prompts["points"][0]
    bbox = [[x1, y1, x2, y2]]
    # Create binary mask using numpy
    height, width = prompts["image"].shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[int(y1) : int(y2), int(x1) : int(x2)] = 1

    # Save mask as image in gradio temp folder
    mask_pil = Image.fromarray(mask * 255)
    mask_pil.save("flagged/mask.jpg")

    print("Finish preparing....")

    generated_image = moMA_main_modal.generate_images(
        "flagged/image.jpg",
        "flagged/mask.jpg",
        subject,
        text_prompt,
        strength=strength,
        seed=seed,
        return_mask=return_mask,
    )
    return generated_image


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # MoMA Text-to-Image Generation
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            ref_image = ImagePrompter(
                show_label=False,
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

            return inference(
                ref_image,
                prompt_text,
                subject,
                strength,
                seed,
            )

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

        # save_btn.click(
        #     fn=save_image,
        #     inputs=[output_image, prompt],
        # )

demo.launch(server_name="0.0.0.0")
