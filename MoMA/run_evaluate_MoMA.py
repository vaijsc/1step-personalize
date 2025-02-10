import os
import torch
from pytorch_lightning import seed_everything
from torchvision.utils import save_image
import warnings

from model_lib.modules import MoMA_main_modal
from model_lib.utils import parse_args

warnings.filterwarnings("ignore")

seed = 42
seed_everything(seed)
args = parse_args()

args.device = torch.device("cuda", 0)


# if you have 22 Gb GPU memory:
args.load_8bit, args.load_4bit = False, False

# if you have 18 Gb GPU memory:
# args.load_8bit, args.load_4bit = True, False

# if you have 14 Gb GPU memory:
# args.load_8bit, args.load_4bit = False, True


# load MoMA from HuggingFace. Auto download
moMA_main_modal = MoMA_main_modal(args).to(args.device, dtype=torch.bfloat16)

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

# reference image and its mask
# rgb_path = "example_images/newImages/can.jpg"
# mask_path = "example_images/newImages/can_mask.jpg"
# subject = "can"

rgb_path = "example_images/newImages/dog.jpg"
mask_path = "example_images/newImages/dog_mask.jpg"
subject = "dog"

prompt_set = [
            "a dog in the jungle",
            "a dog in the snow",
            "a dog on the beach",
            "a dog on a cobblestone street",
            "a dog on top of pink fabric",
            "a dog on top of a wooden floor",
            "a dog with a city in the background",
            "a dog with a mountain in the background",
            "a dog with a blue house in the background",
            "a dog on top of a purple rug in a forest",
            "a dog wearing a red hat",
            "a dog wearing a santa hat",
            "a dog wearing a rainbow scarf",
            "a dog wearing a black top hat and a monocle",
            "a dog in a chef outfit",
            "a dog in a firefighter outfit",
            "a dog in a police outfit",
            "a dog wearing pink glasses",
            "a dog wearing a yellow shirt",
            "a dog in a purple wizard outfit",
            "a red dog",
            "a purple dog",
            "a shiny dog",
            "a wet dog",
            "a cube shaped dog"
        ]

# Let's generate new images!
# prompt_set = [
#     "a can in the jungle",
#     "a can in the snow",
#     "a can on the beach",
#     "a can on a cobblestone street",
#     "a can on top of pink fabric",
#     "a can on top of a wooden floor",
#     "a can with a city in the background",
#     "a can with a mountain in the background",
#     "a can with a blue house in the background",
#     "a can on top of a purple rug in a forest",
#     "a can with a wheat field in the background",
#     "a can with a tree and autumn leaves in the background",
#     "a can with the Eiffel Tower in the background",
#     "a can floating on top of water",
#     "a can floating in an ocean of milk",
#     "a can on top of green grass with sunflowers around it",
#     "a can on top of a mirror",
#     "a can on top of the sidewalk in a crowded street",
#     "a can on top of a dirt road",
#     "a can on top of a white rug",
#     "a red can",
#     "a purple can",
#     "a shiny can",
#     "a wet can",
#     "a cube shaped can",
# ]

################ change context ##################
for prompt in prompt_set:
    generated_image = moMA_main_modal.generate_images(
        rgb_path, mask_path, subject, prompt, strength=1.5, seed=seed, return_mask=False
    )  # set strength to 1.0 for more accurate details
    generated_image.save(f"{args.output_path}/{subject}_{prompt}.jpg")

################ change texture ##################
# prompt = "A wooden sculpture of a car on the table."
# generated_image = moMA_main_modal.generate_images(
#     rgb_path, mask_path, subject, prompt, strength=0.4, seed=4, return_mask=True
# )  # set strength to 0.4 for better prompt fidelity
# save_image(generated_image, f"{args.output_path}/{subject}_{prompt}.jpg")
