import torch

arr = list(range(1000, 27100, 1000))


for i in arr:
    name = f"checkpoint-{i}"
    ckpt = f"distill_ipa_output/{name}/pytorch_model.bin"

    sd = torch.load(ckpt, map_location="cpu")
    image_proj_sd = {}
    ip_sd = {}
    unet = {}
    for k in sd:
        if k.startswith("unet"):
            unet[k.replace("unet.", "")] = sd[k]
        elif k.startswith("image_proj_model"):
            image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
        elif k.startswith("adapter_modules"):
            ip_sd[k.replace("adapter_modules.", "")] = sd[k]

    torch.save(
        {"image_proj": image_proj_sd, "ip_adapter": ip_sd},
        f"distilled/dmd2-ip_adapter_{name}_1.bin",
    )
    torch.save({"unet": unet}, f"distilled/unet_{name}.bin")
