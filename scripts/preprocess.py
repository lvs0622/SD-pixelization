import argparse
import os
from typing import Tuple
from PIL import Image

from utils.image_ops import load_image, save_image, center_crop, normalize, edge_canny
from utils.pixel import quantize_colors

# 数据预处理：裁剪、归一化、颜色量化与边缘提取


def process_image(
    input_path: str,
    output_dir: str,
    crop_size: Tuple[int, int],
    colors: int,
    save_edge: bool = True,
):
    img = load_image(input_path)
    img = center_crop(img, crop_size)
    img = normalize(img)
    img = quantize_colors(img, colors)

    base = os.path.splitext(os.path.basename(input_path))[0]
    out_img_path = os.path.join(output_dir, "images", base + ".jpg")
    save_image(img, out_img_path)

    if save_edge:
        edge = edge_canny(img)
        out_edge_path = os.path.join(output_dir, "edges", base + "_edge.jpg")
        save_image(edge, out_edge_path)


def run_dir(input_dir: str, output_dir: str, crop: int, colors: int, save_edge: bool):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "edges"), exist_ok=True)
    for fn in os.listdir(input_dir):
        ip = os.path.join(input_dir, fn)
        if not os.path.isfile(ip):
            continue
        ext = os.path.splitext(ip)[1].lower()
        if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            continue
        process_image(ip, output_dir, (crop, crop), colors, save_edge)


def build_parser():
    p = argparse.ArgumentParser(description="数据预处理脚本")
    p.add_argument("--input_dir", type=str, required=True, help="原始数据目录")
    p.add_argument("--output_dir", type=str, required=True, help="输出目录")
    p.add_argument("--crop", type=int, default=512, help="中心裁剪尺寸")
    p.add_argument("--colors", type=int, default=32, help="量化颜色数")
    p.add_argument("--save_edge", action="store_true", help="是否保存边缘图")
    return p


def main():
    args = build_parser().parse_args()
    run_dir(args.input_dir, args.output_dir, args.crop, args.colors, args.save_edge)


if __name__ == "__main__":
    main()
