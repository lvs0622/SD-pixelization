from PIL import Image

# 像素化与颜色量化工具

def pixelate(img: Image.Image, block_size: int) -> Image.Image:
    """按块像素化，最近邻缩放"""
    block_size = max(1, int(block_size))
    w, h = img.size
    down_w = max(1, w // block_size)
    down_h = max(1, h // block_size)
    small = img.resize((down_w, down_h), Image.NEAREST)
    return small.resize((w, h), Image.NEAREST)


def quantize_colors(img: Image.Image, num_colors: int) -> Image.Image:
    """颜色量化，减少色彩数"""
    num_colors = max(2, int(num_colors))
    # 使用 PIL 自带的量化方法
    q = img.convert("P", palette=Image.ADAPTIVE, colors=num_colors)
    return q.convert("RGB")
