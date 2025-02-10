import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


def generate_prior_images(
    prompt, num_images, output_dir, model_id="runwayml/stable-diffusion-v1-5"
):
    """
    Generate multiple images from a prompt using Stable Diffusion

    Args:
        prompt (str): The text prompt to generate images from
        num_images (int): Number of images to generate
        output_dir (str): Directory to save the generated images
        model_id (str): The model ID to use for generation
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    # Generate images
    for i in range(num_images):
        image = pipe(prompt).images[0]

        # Save the image
        image_path = os.path.join(output_dir, f"prior_gen_{i}.png")
        image.save(image_path)

    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Example usage
    prompt = "a photo of a can"
    num_images = 100
    output_dir = "priors/can/"

    generate_prior_images(prompt, num_images, output_dir)
