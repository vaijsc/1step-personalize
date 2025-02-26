
import argparse
import gc
import glob
import logging
import math
import os
import random
import shutil
from pathlib import Path

import pdb
import itertools
import copy

import accelerate

import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

from dataloaders import get_dict_data
from datasets import load_dataset, Image, Dataset, Features
from PIL import Image as Image_PIL
from DISTS_pytorch import DISTS
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import to_pil_image

from typing import Tuple, Union, Optional, List

from torchvision.transforms.functional import pil_to_tensor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torchvision import transforms
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

from transformers import CLIPImageProcessor
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

logger = get_logger(__name__)

class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            print("Loading IP Adapter from:", ckpt_path)
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

    def set_scale(self, scale):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")
    
T = torch.Tensor
TN = Optional[T]
TS = Union[Tuple[T, ...], List[T]]

def gen_random_tensor_fix_seed(input_shape, seed, weight_dtype, device):
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Ensure that all operations are deterministic on the current device
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Generate the random tensor
    return torch.randn(*input_shape, dtype=weight_dtype, device=device)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_sb_generator",
        type=str,
        default="../swiftbrush-model/unet_ema",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X steps. The validation process consists of running the prompts"
            " `args.validation_prompts` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="swiftbrush-output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--weight_dtype",
        type=str,
        default="fp32",
        help=(
            "data type weight. Default as fp32"
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use for unet.",
    )
    # parser.add_argument(
    #     "--learning_rate_zconv",
    #     type=float,
    #     default=1e-4,
    #     help="Initial learning rate (after the potential warmup period) to use for lora teacher.",
    # )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    # parser.add_argument(
    #     "--lr_scheduler_zconv",
    #     type=str,
    #     default="constant",
    #     help=(
    #         'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
    #         ' "constant", "constant_with_warmup"]'
    #     ),
    # )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="The classifier-free guidance scale.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_false", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )

    parser.add_argument(
        "--task",
        type=str,
        default='general',
        help=("Training dataset"),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

# Adapted from pipelines.StableDiffusionPipeline.encode_prompt
def encode_prompt(prompts, text_encoder, tokenizer, is_train=True):
    captions = []
    for caption in prompts:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
        )[0]

    return {"prompt_embeds": prompt_embeds.cpu()}

def load_512(image_path: str, left=0, right=0, top=0, bottom=0):
    image = np.array(Image_PIL.open(image_path))[:, :, :3]    
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image_PIL.fromarray(image).resize((512, 512)))
    return image

@torch.no_grad()
def inference(vae, tokenizer, text_encoder,
              noise_scheduler, sb_ip_adapter_generator, 
              image_encoder,
              prompt, image_path, clip_image_processor,
              weight_dtype, device):
    
    # prepare stuff
    T = torch.ones((1,), dtype=torch.int64, device="cuda")
    T = T * (noise_scheduler.config.num_train_timesteps - 1)
    alphas_cumprod = noise_scheduler.alphas_cumprod.to("cuda")
    alpha_t = (alphas_cumprod[T] ** 0.5).view(-1, 1, 1, 1)
    sigma_t = ((1 - alphas_cumprod[T]) ** 0.5).view(-1, 1, 1, 1)
    del alphas_cumprod
    last_timestep = torch.ones((1,), dtype=torch.int64, device=device) * 999

    # Encoded latent
    src_img = load_512(image_path)
    src_img = torch.from_numpy(src_img).float().permute(2, 0, 1) / 127.5 - 1
    src_img = src_img.unsqueeze(0).to(device, dtype=weight_dtype)
    
    latents = vae.encode(src_img.to(device, dtype=weight_dtype)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    # Sampling noise
    input_shape = (1, 4, args.resolution // 8, args.resolution // 8)
    noise = torch.randn(input_shape, dtype=weight_dtype, device=device)

    noisy_latents = alpha_t * latents + sigma_t * noise
    # noise = gen_random_tensor_fix_seed(input_shape, args.seed,weight_dtype, device)
    
    # Prepare prompt embeds
    input_id = tokenize_captions(prompt, tokenizer).to(device)
    prompt_embeds = text_encoder(input_id)[0].to(device, dtype=weight_dtype)

    # Get image embeds for IP-Adapter
    pil_ref_image = [Image_PIL.open(image_path)]
                    
    clip_images = [clip_image_processor(images=sample, return_tensors="pt").pixel_values for sample in pil_ref_image]
    clip_images = torch.cat(clip_images, dim=0)
    image_embeds = image_encoder(clip_images.to(device, dtype=weight_dtype)).image_embeds
            
    
    # Get denoised image with one step image generator (sb in this case) with ip-condition (scale=1)
    noise_pred = sb_ip_adapter_generator(noisy_latents, last_timestep, prompt_embeds, image_embeds)
    pred_latents = (noisy_latents - sigma_t * noise_pred) / alpha_t
    
    out_image = pred_latents / vae.config.scaling_factor
    out_image = (
        vae.decode(out_image.to(dtype=weight_dtype)).sample.float() + 1
    ) / 2

    # Get denoised image with one step image generator (sb in this case) with ip-condition (scale=0)
    sb_ip_adapter_generator.set_scale(0)
    noise_pred = sb_ip_adapter_generator(noisy_latents, last_timestep, prompt_embeds, image_embeds)
    pred_latents = (noisy_latents - sigma_t * noise_pred) / alpha_t
    out_image_scale0 = pred_latents / vae.config.scaling_factor
    out_image_scale0 = (
        vae.decode(out_image_scale0.to(dtype=weight_dtype)).sample.float() + 1
    ) / 2

    # set scale back to 1
    sb_ip_adapter_generator.set_scale(1)
    
    return out_image.detach().cpu()[0], out_image_scale0.detach().cpu()[0]

def tokenize_captions(captions, tokenizer):
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    logger.info(weight_dtype)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False
    )

    # import correct text encoder classes
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)

    logger.info(args.pretrained_sb_generator)

    unet_gen_ip = UNet2DConditionModel.from_pretrained(args.pretrained_sb_generator).to(
        accelerator.device,
        dtype=weight_dtype
    )
    unet_gen_ip.eval()
    unet_gen_ip.requires_grad_(False)
    
    # ip adapter
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path, subfolder="models/image_encoder")
    image_encoder.requires_grad_(False)
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet_gen_ip.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    )
    
    # init adapter modules
    attn_procs = {}
    unet_sd = unet_gen_ip.state_dict()
    for name in unet_gen_ip.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet_gen_ip.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet_gen_ip.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet_gen_ip.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet_gen_ip.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
    unet_gen_ip.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet_gen_ip.attn_processors.values())
    
    sb_ip_adapter_generator = IPAdapter(unet_gen_ip, image_proj_model, adapter_modules)

    # Freeze vae, text encoders and teacher.
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to(accelerator.device, dtype=torch.float32)
    vae.enable_slicing()
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    
    clip_image_processor = CLIPImageProcessor()

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    if isinstance(model, type(accelerator.unwrap_model(sb_ip_adapter_generator))):
                        torch.save(model.state_dict(), os.path.join(output_dir, "ip_adapter.bin"))
                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                if isinstance(model, type(accelerator.unwrap_model(sb_ip_adapter_generator))):
                    ckpt_ip_path= os.path.join(input_dir, "ip_adapter.bin")
                    model.load_state_dict(torch.load(ckpt_ip_path))
                    
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

        args.learning_rate_lora = (
            args.learning_rate_lora
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
        # optimizer_class = torch.optim.SGD

    # Optimizer creation
    params = [sb_ip_adapter_generator.image_proj_model.parameters(),
              sb_ip_adapter_generator.adapter_modules.parameters()]
    
    # adam
    optimizer = optimizer_class(
        itertools.chain(*params),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Prepare dataset 
    train_dict_data, val_dict_data = get_dict_data(task=args.task)
    dataset = Dataset.from_dict(train_dict_data)
    dataset = dataset.cast_column("ref_image", Image())
    dataset = dataset.cast_column("out_image", Image())
    
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset.column_names
    ref_image_column, out_image_column, text_column = column_names
    
    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, text_column, is_train=True):
        captions = []
        for caption in examples[text_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{text_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    
    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        ref_images = [image.convert("RGB") for image in examples[ref_image_column]]
        out_images = [image.convert("RGB") for image in examples[out_image_column]]
        examples["ref_pixel_values"] = [train_transforms(image) for image in ref_images]
        examples["out_pixel_values"] = [train_transforms(image) for image in out_images]
        examples["input_ids"] = tokenize_captions(examples, 'text')
        examples["text"] = [text for text in examples["text"]]
        return examples
    
    with accelerator.main_process_first():
        train_dataset = dataset.with_transform(preprocess_train)

    def collate_fn(examples):
        ref_pixel_values = torch.stack([example["ref_pixel_values"] for example in examples])
        ref_pixel_values = ref_pixel_values.to(memory_format=torch.contiguous_format).float()

        out_pixel_values = torch.stack([example["out_pixel_values"] for example in examples])
        out_pixel_values = out_pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = torch.stack([example["input_ids"] for example in examples])
        text = [example["text"] for example in examples]
        return {"ref_pixel_values": ref_pixel_values, 
                "out_pixel_values": out_pixel_values,
                "input_ids": input_ids,
                "text": text}
                
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    
    sb_ip_adapter_generator, optimizer, train_dataloader, lr_scheduler = (
        accelerator.prepare(
            sb_ip_adapter_generator, optimizer, train_dataloader, lr_scheduler
        )
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers("train_personalize", config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Get predefined alphas cummulative product for sb generator
    T = torch.ones((1,), dtype=torch.int64, device="cuda")
    T = T * (noise_scheduler.config.num_train_timesteps - 1)
    alphas_cumprod = noise_scheduler.alphas_cumprod.to("cuda")
    alpha_t = (alphas_cumprod[T] ** 0.5).view(-1, 1, 1, 1)
    sigma_t = ((1 - alphas_cumprod[T]) ** 0.5).view(-1, 1, 1, 1)
    del alphas_cumprod

    last_timestep = torch.ones((1,), dtype=torch.int64, device=accelerator.device) * 999
    
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss_l2 = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(sb_ip_adapter_generator):
                # Get GT image latent
                with torch.no_grad():
                    gt_latents = vae.encode(batch["out_pixel_values"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    gt_latents = gt_latents * vae.config.scaling_factor    

                # Sampling noise
                noise = torch.randn_like(gt_latents)
                # noise = gen_random_tensor_fix_seed(gt_latents.shape, args.seed,weight_dtype, accelerator.device)

                # add noise to create noisy latent
                noisy_latents = alpha_t * gt_latents + sigma_t * noise
                
                # Prepare prompt embeds
                prompt_embeds = text_encoder(batch["input_ids"], return_dict=False)[0].to(accelerator.device, dtype=weight_dtype)
                
                # Get image embeds for IP-Adapter
                ref_image = (batch["ref_pixel_values"] + 1) / 2
                pil_ref_image = [to_pil_image((sample * 255).clamp(0, 255).to(torch.uint8)) for sample in ref_image]
                                
                clip_images = [clip_image_processor(images=sample, return_tensors="pt").pixel_values for sample in pil_ref_image]
                clip_images = torch.cat(clip_images, dim=0)
                image_embeds = image_encoder(clip_images.to(accelerator.device, dtype=weight_dtype)).image_embeds
            
                # Get denoised image with one step image generator (sb in this case)
                noise_pred = sb_ip_adapter_generator(noisy_latents, last_timestep, prompt_embeds, image_embeds)
                
                ## L2 Loss
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss_l2 += avg_loss.item() / args.gradient_accumulation_steps
                
                # Backpropagate
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(sb_ip_adapter_generator.image_proj_model.parameters(), args.max_grad_norm)
                    accelerator.clip_grad_norm_(sb_ip_adapter_generator.adapter_modules.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step() 
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {
                        "train_loss_l2": train_loss_l2,
                    }, step=global_step
                )
                
                train_loss_l2 = 0.0

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0 or global_step == args.max_train_steps:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if global_step % args.validation_steps == 0 or global_step == args.max_train_steps:
                    logger.info(
                        "Running validation... \Reconstruct images"
                    )

                    # run inference
                    with torch.cuda.amp.autocast():
                        out_image_stores = {}
                        out_image_stores_scale0 = {}
                        
                        # on train set
                        log_image_test = {"train": train_dict_data,
                                          "val": val_dict_data}
                        
                        for split in log_image_test:
                            dict_data = log_image_test[split]
                            len_samples = len(dict_data["ref_image"][:20]) # first 20 samples
                            for idx in range(len_samples):
                                text = dict_data["text"][idx]
                                inp_img_path = dict_data["ref_image"][idx]
                                
                                if split == "train":
                                    key_reconstruct_image = f"train_samples_{text}"
                                else:
                                    key_reconstruct_image = f"val_samples_{text}"
                                
                                out_image, out_image_scale0 = inference(
                                    vae,
                                    tokenizer,
                                    text_encoder,
                                    noise_scheduler,
                                    sb_ip_adapter_generator,
                                    image_encoder,
                                    text,
                                    inp_img_path,
                                    clip_image_processor,
                                    weight_dtype,
                                    accelerator.device
                                )
                                
                                out_image_stores[key_reconstruct_image] = out_image
                                out_image_stores_scale0[key_reconstruct_image] = out_image_scale0
                    for tracker in accelerator.trackers:
                        for split in log_image_test:
                            dict_data = log_image_test[split]
                            len_samples = len(dict_data["ref_image"][:20])
                            
                            for idx in range(len_samples):
                                text = dict_data["text"][idx]
                                gt_out_img = pil_to_tensor(
                                    Image_PIL.open(dict_data["out_image"][idx])
                                        .convert("RGB").resize((512, 512))
                                )/255

                                ref_img = pil_to_tensor(
                                    Image_PIL.open(dict_data["ref_image"][idx])
                                        .convert("RGB").resize((512, 512))
                                )/255
                                
                                if split == "train":
                                    key_reconstruct_image = f"train_samples_{text}"
                                else:
                                    key_reconstruct_image = f"val_samples_{text}"
                                                                
                                vis_imgs = torch.cat([ref_img,
                                                    gt_out_img,
                                                    out_image_stores[key_reconstruct_image],
                                                    out_image_stores_scale0[key_reconstruct_image]], dim=2)
                                tracker.writer.add_images(
                                    f"ID:{key_reconstruct_image}", np.asarray(vis_imgs), global_step, dataformats="CHW"
                                )
                
            logs = {
                "step_loss_l2": loss.detach().item(),
                "lr_unet": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()
    
if __name__ == "__main__":
    args = parse_args()
    main(args)

