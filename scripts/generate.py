import argparse
import json
import os
from typing import Optional, Tuple, List, Dict
from PIL import Image
import numpy as np
import torch
from diffusers.utils import logging as dlogging

from utils.image_ops import load_image, save_image, edge_canny
from utils.pixel import pixelate, quantize_colors
from models.pipeline_loader import load_sd_controlnet_pipeline

# 生成与批量处理脚本

PIPELINE_CACHE: Dict[str, any] = {}
dlogging.set_verbosity_error()


def _pipeline_key(
    base_model_id: str, controlnet_id: str, lora_path: Optional[str]
) -> str:
    return f"{base_model_id}|{controlnet_id}|{lora_path or 'none'}"


def get_pipeline(
    base_model_id: str,
    controlnet_id: str,
    lora_path: Optional[str],
    device: Optional[str] = None,
):
    key = _pipeline_key(base_model_id, controlnet_id, lora_path)
    if key not in PIPELINE_CACHE:
        PIPELINE_CACHE[key] = load_sd_controlnet_pipeline(
            base_model_id=base_model_id,
            controlnet_id=controlnet_id,
            lora_path=lora_path,
            device=device,
        )
    return PIPELINE_CACHE[key]


def create_grid(images: List[Image.Image], cols: int = 2) -> Image.Image:
    """拼接对比图网格"""
    if not images:
        raise ValueError("no images for grid")
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (w * cols, h * rows))
    for idx, im in enumerate(images):
        r = idx // cols
        c = idx % cols
        grid.paste(im, (c * w, r * h))
    return grid


def generate_pixel_style_image(
    input_img: Image.Image,
    base_model_id: str = "runwayml/stable-diffusion-v1-5",
    controlnet_id: str = "lllyasviel/sd-controlnet-canny",
    lora_path: Optional[str] = None,
    prompt: str = "",
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    controlnet_conditioning_scale: float = 1.0,
    pixel_block_size: int = 8,
    color_count: int = 16,
    seed: Optional[int] = None,
    device: Optional[str] = None,
) -> Image.Image:
    """单张图像的像素风格生成流程"""
    pipe = get_pipeline(base_model_id, controlnet_id, lora_path, device=device)
    cond = edge_canny(input_img)
    generator = torch.Generator(device=pipe.device).manual_seed(seed) if seed else None

    out = pipe(
        prompt=prompt or "",
        image=cond,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        generator=generator,
    ).images[0]

    out = pixelate(out, pixel_block_size)
    out = quantize_colors(out, color_count)
    return out


def infer_lora_path(style_name: Optional[str]) -> Optional[str]:
    """根据风格名推断 LoRA 权重路径"""
    if not style_name:
        return None
    base_dir = os.path.join("models", "lora", style_name)
    candidates = []
    if os.path.isdir(base_dir):
        for fn in os.listdir(base_dir):
            if fn.endswith(".safetensors") or fn.endswith(".bin"):
                candidates.append(os.path.join(base_dir, fn))
    return candidates[0] if candidates else None


def run_single(
    input_path: str,
    output_path: str,
    prompt: str,
    base_model_id: str,
    controlnet_id: str,
    style_name: Optional[str],
    num_inference_steps: int,
    guidance_scale: float,
    controlnet_conditioning_scale: float,
    pixel_block_size: int,
    color_count: int,
    seed: Optional[int],
    device: Optional[str],
) -> Tuple[str, str]:
    """处理单张输入并保存结果与对比图"""
    img = load_image(input_path)
    lora_path = infer_lora_path(style_name)
    out = generate_pixel_style_image(
        input_img=img,
        base_model_id=base_model_id,
        controlnet_id=controlnet_id,
        lora_path=lora_path,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        pixel_block_size=pixel_block_size,
        color_count=color_count,
        seed=seed,
        device=device,
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_image(out, output_path)

    grid = create_grid([img, out], cols=2)
    grid_path = os.path.splitext(output_path)[0] + "_grid.jpg"
    save_image(grid, grid_path)

    meta = {
        "input": input_path,
        "output": output_path,
        "prompt": prompt,
        "base_model_id": base_model_id,
        "controlnet_id": controlnet_id,
        "style_name": style_name,
        "lora_path": lora_path,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "controlnet_conditioning_scale": controlnet_conditioning_scale,
        "pixel_block_size": pixel_block_size,
        "color_count": color_count,
        "seed": seed,
    }
    meta_path = os.path.splitext(output_path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return output_path, grid_path


def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def run_batch(
    input_dir: str,
    output_dir: str,
    prompt: str,
    base_model_id: str,
    controlnet_id: str,
    style_name: Optional[str],
    num_inference_steps: int,
    guidance_scale: float,
    controlnet_conditioning_scale: float,
    pixel_block_size: int,
    color_count: int,
    seed: Optional[int],
    device: Optional[str],
):
    """批量处理目录内所有图像文件"""
    os.makedirs(output_dir, exist_ok=True)
    for fn in os.listdir(input_dir):
        ip = os.path.join(input_dir, fn)
        if not is_image_file(ip):
            continue
        op = os.path.join(output_dir, os.path.splitext(fn)[0] + "_pixel.jpg")
        run_single(
            input_path=ip,
            output_path=op,
            prompt=prompt,
            base_model_id=base_model_id,
            controlnet_id=controlnet_id,
            style_name=style_name,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            pixel_block_size=pixel_block_size,
            color_count=color_count,
            seed=seed,
            device=device,
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="像素风格迁移生成脚本")
    p.add_argument("--input", type=str, required=True, help="输入图像或目录")
    p.add_argument("--output", type=str, required=True, help="输出图像或目录")
    p.add_argument("--prompt", type=str, default="", help="文本提示")
    p.add_argument(
        "--base_model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="SD 基础模型",
    )
    p.add_argument(
        "--controlnet_id",
        type=str,
        default="lllyasviel/sd-controlnet-canny",
        help="ControlNet 模型",
    )
    p.add_argument("--style", type=str, default=None, help="风格模板名，对应 models/lora/<name>")
    p.add_argument("--steps", type=int, default=30, help="采样步数")
    p.add_argument("--guidance", type=float, default=7.5, help="风格强度（guidance scale）")
    p.add_argument(
        "--structure",
        type=float,
        default=1.0,
        help="结构保留程度（controlnet conditioning scale）",
    )
    p.add_argument("--block", type=int, default=8, help="像素块大小")
    p.add_argument("--colors", type=int, default=16, help="色彩数量")
    p.add_argument("--seed", type=int, default=None, help="随机种子")
    p.add_argument("--device", type=str, default=None, help="设备，例如 cuda 或 cpu")
    return p


def main():
    args = build_parser().parse_args()
    input_path = args.input
    output_path = args.output

    if os.path.isdir(input_path):
        run_batch(
            input_dir=input_path,
            output_dir=output_path,
            prompt=args.prompt,
            base_model_id=args.base_model_id,
            controlnet_id=args.controlnet_id,
            style_name=args.style,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            controlnet_conditioning_scale=args.structure,
            pixel_block_size=args.block,
            color_count=args.colors,
            seed=args.seed,
            device=args.device,
        )
    else:
        op = output_path
        if os.path.isdir(output_path):
            base = os.path.basename(input_path)
            op = os.path.join(output_path, os.path.splitext(base)[0] + "_pixel.jpg")
        run_single(
            input_path=input_path,
            output_path=op,
            prompt=args.prompt,
            base_model_id=args.base_model_id,
            controlnet_id=args.controlnet_id,
            style_name=args.style,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            controlnet_conditioning_scale=args.structure,
            pixel_block_size=args.block,
            color_count=args.colors,
            seed=args.seed,
            device=args.device,
        )


if __name__ == "__main__":
    main()
