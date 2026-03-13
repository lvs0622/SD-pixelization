import gradio as gr
import torch
import numpy as np
from PIL import Image
from generate1 import PixelArtGenerator  # 导入你现有的类
import sys
import os

# 将当前脚本所在的目录添加到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
# ... 其他导入



# 1. 实例化你现有的生成引擎
# 建议在初始化时指定好路径
base_dir = "/root/autodl-tmp/SD-pixelization-2"
gen_engine = PixelArtGenerator(
    base_model_path=f"{base_dir}/models/sd15",
    lora_dir=f"{base_dir}/models/lora/1bit", 
    controlnet_path=f"{base_dir}/models/controlnet_canny"
)

def apply_palette(image, palette_type, custom_colors=None):
    """
    调色盘处理逻辑：将生成的黑白像素映射到用户定义的颜色
    """
    if palette_type == "Original (B&W)":
        return image
    
    # 将图像转为灰度并二值化确定 mask
    img_bw = image.convert("L")
    mask = np.array(img_bw) > 127
    
    # 获取颜色
    if palette_type == "Retro Game (Green)":
        color_bg, color_fg = [15, 56, 15], [155, 188, 15]  # Gameboy 风格
    elif palette_type == "Cyberpunk":
        color_bg, color_fg = [20, 0, 40], [255, 0, 255]
    else: # Custom
        # custom_colors 格式为 "#RRGGBB"
        def hex_to_rgb(h): return [int(h[i:i+2], 16) for i in (1, 3, 5)]
        color_bg = hex_to_rgb(custom_colors[0])
        color_fg = hex_to_rgb(custom_colors[1])

    # 应用颜色映射
    new_img = np.zeros((image.size[1], image.size[0], 3), dtype=np.uint8)
    new_img[mask] = color_fg
    new_img[~mask] = color_bg
    return Image.fromarray(new_img)

def inference_wrapper(input_img, style_type, palette_type, color_bg, color_fg, strength, cn_scale):
    if style_type != "1-bit (Native)":
        return None, f"抱歉，{style_type} 风格模型正在训练中，目前仅支持 1-bit 测试。"
    
    # 1. 保存输入图片供模型读取
    input_img.save("web_input.png")
    
    # === 关键修改 ===
    # 注意：gen_engine.generate 返回一个元组 (raw_result, processed_result)
    # 你的代码原意是用 '_' 忽略了后处理结果，现在我们需要它
    # '_' 原来是这样: _, processed_output = gen_engine.generate(...)
    
    # 2. 调用生成逻辑，显式获取后处理后的 1-bit 图像
    # 我们不再需要原始的灰色图像 'raw_output' 进行调色
    _, processed_output = gen_engine.generate(
        image_path="web_input.png",
        strength=strength,
        controlnet_scale=cn_scale,
        guidance_scale=12.0
    )
    
    # 3. 将经过正确 1-bit 后处理的图像传给调色盘应用函数
    final_output = apply_palette(processed_output, palette_type, [color_bg, color_fg])
    
    return final_output, "生成成功！"
# --- Gradio UI 布局 ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 像素风格迁移研究演示系统 (Pixel Style Transfer)")
    gr.Markdown("本项目基于扩散模型干预技术，实现了从普通图像到原生像素艺术的风格迁移。")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="上传原图")
            
            with gr.Group():
                gr.Markdown("### 风格配置")
                style_radio = gr.Radio(
                    ["1-bit (Native)", "8-bit (Planning)", "16-bit (Planning)"], 
                    value="1-bit (Native)", 
                    label="目标风格"
                )
                
            with gr.Group():
                gr.Markdown("### 调色盘自定义")
                palette_radio = gr.Radio(
                    ["Original (B&W)", "Retro Game (Green)", "Cyberpunk", "Custom"], 
                    value="Original (B&W)", 
                    label="预设方案"
                )
                with gr.Row():
                    bg_color = gr.ColorPicker(value="#000000", label="背景色 (Dark)")
                    fg_color = gr.ColorPicker(value="#FFFFFF", label="前景色 (Light)")
            
            with gr.Accordion("高级生成参数", open=False):
                strength_slider = gr.Slider(0.1, 1.0, value=0.75, label="迁移强度 (Strength)")
                cn_slider = gr.Slider(0.0, 1.0, value=0.8, label="结构保留度 (ControlNet)")
            
            run_btn = gr.Button("开始迁移", variant="primary")

        with gr.Column(scale=1):
            output_img = gr.Image(label="生成结果", interactive=False)
            status_text = gr.Textbox(label="系统状态")
            save_btn = gr.Button("💾 保存结果 (右键图片即可下载)")

    # 交互逻辑
    run_btn.click(
        fn=inference_wrapper,
        inputs=[input_img, style_radio, palette_radio, bg_color, fg_color, strength_slider, cn_slider],
        outputs=[output_img, status_text]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006)
