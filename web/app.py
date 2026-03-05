import gradio as gr
import os
from PIL import Image

from scripts.generate import generate_pixel_style_image, infer_lora_path

# 简易 Gradio 前端：上传原图/选择风格/可调参数/显示结果


def list_styles() -> list:
    styles_dir = os.path.join("models", "lora")
    styles = []
    if os.path.isdir(styles_dir):
        for name in os.listdir(styles_dir):
            p = os.path.join(styles_dir, name)
            if os.path.isdir(p):
                styles.append(name)
    styles.sort()
    return ["None"] + styles


def generate_ui(
    img: Image.Image,
    style_name: str,
    prompt: str,
    steps: int,
    guidance: float,
    structure: float,
    block: int,
    colors: int,
    seed: int,
    base_model_id: str,
    controlnet_id: str,
):
    style = None if style_name == "None" else style_name
    out = generate_pixel_style_image(
        input_img=img,
        base_model_id=base_model_id,
        controlnet_id=controlnet_id,
        lora_path=infer_lora_path(style),
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        controlnet_conditioning_scale=structure,
        pixel_block_size=block,
        color_count=colors,
        seed=seed if seed != -1 else None,
        device=None,
    )
    return out


with gr.Blocks() as demo:
    gr.Markdown("# 像素风格迁移（Stable Diffusion + ControlNet + LoRA）")
    with gr.Row():
        with gr.Column():
            img = gr.Image(type="pil", label="上传原图")
            style = gr.Dropdown(choices=list_styles(), label="选择风格模板/LoRA", value="None")
            prompt = gr.Textbox(label="文本提示", value="")
            steps = gr.Slider(1, 60, value=30, step=1, label="采样步数")
            guidance = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="风格强度")
            structure = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="结构保留程度")
            block = gr.Slider(1, 64, value=8, step=1, label="像素块大小")
            colors = gr.Slider(2, 64, value=16, step=1, label="色彩数量")
            seed = gr.Number(value=-1, label="随机种子（-1为随机）")
            base_model_id = gr.Textbox(value="runwayml/stable-diffusion-v1-5", label="SD 模型ID")
            controlnet_id = gr.Textbox(value="lllyasviel/sd-controlnet-canny", label="ControlNet 模型ID")
            btn = gr.Button("生成")
        with gr.Column():
            out = gr.Image(label="生成结果")
    btn.click(
        fn=generate_ui,
        inputs=[img, style, prompt, steps, guidance, structure, block, colors, seed, base_model_id, controlnet_id],
        outputs=[out],
    )

if __name__ == "__main__":
    demo.launch()
