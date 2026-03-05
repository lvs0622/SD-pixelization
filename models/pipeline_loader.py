import torch
from typing import Optional
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# Stable Diffusion + ControlNet + LoRA 加载器

def _select_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sd_controlnet_pipeline(
    base_model_id: str = "runwayml/stable-diffusion-v1-5",
    controlnet_id: str = "lllyasviel/sd-controlnet-canny",
    lora_path: Optional[str] = None,
    device: Optional[str] = None,
):
    """加载带 ControlNet 的 SD 管道，支持可选 LoRA"""
    dev = _select_device(device)
    dtype = torch.float16 if dev.type == "cuda" else torch.float32

    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_id, controlnet=controlnet, torch_dtype=dtype
    )

    if lora_path:
        try:
            pipe.load_lora_weights(lora_path)
        except Exception:
            pass

    pipe = pipe.to(dev)
    return pipe
