import os
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import cv2

# 基本图像读写与边缘提取工具

def load_image(path: str) -> Image.Image:
    """加载图像为RGB"""
    img = Image.open(path).convert("RGB")
    return img


def save_image(img: Image.Image, path: str) -> None:
    """保存图像，自动创建目录"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def to_numpy(img: Image.Image) -> np.ndarray:
    """PIL 转 Numpy，RGB"""
    return np.array(img)


def from_numpy(arr: np.ndarray) -> Image.Image:
    """Numpy 转 PIL，RGB"""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def edge_canny(
    img: Image.Image,
    threshold1: int = 100,
    threshold2: int = 200,
) -> Image.Image:
    """Canny 边缘提取，输出单通道边缘图作为 ControlNet 条件"""
    arr = to_numpy(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return from_numpy(edges_rgb)


def center_crop(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """中心裁剪到指定尺寸"""
    w, h = img.size
    tw, th = size
    left = max(0, (w - tw) // 2)
    top = max(0, (h - th) // 2)
    right = min(w, left + tw)
    bottom = min(h, top + th)
    return img.crop((left, top, right, bottom))


def normalize(img: Image.Image) -> Image.Image:
    """归一化到 [0,255] 的RGB图像"""
    arr = to_numpy(img).astype(np.float32)
    arr = np.clip(arr, 0, 255)
    return from_numpy(arr.astype(np.uint8))
