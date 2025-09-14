import requests
import base64
import os
from PIL import Image
from io import BytesIO
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-91a54db58fc64524b68957de65517a57")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Stable Diffusion API 配置
SD_API_URL = "http://127.0.0.1:7860/sdapi/v1/txt2img"


models_url = "http://127.0.0.1:7860/sdapi/v1/sd-models"
response = requests.get(models_url)
models = [model["model_name"] for model in response.json()]
print("可用模型:", models)

model_name = "ghostmix_v20Bakedvae.safetensors"
choose_model_name = input("请输入模型的名字：（直接回车为默认）")
options_url = "http://127.0.0.1:7860/sdapi/v1/options"
if choose_model_name == '':
    payload = {"sd_model_checkpoint": model_name}
else:payload = {"sd_model_checkpoint": choose_model_name}
requests.post(options_url, json=payload)

current_model = requests.get(options_url).json()["sd_model_checkpoint"]
print(f"当前模型已切换为: {current_model}")


def generate_prompt(natural_language):
    """使用 DeepSeek API 将自然语言转换为优化提示词"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    # 系统提示 - 指导模型如何优化提示词
    system_prompt = """
    你是一个专业的AI图像生成提示词工程师。请将用户输入的自然语言描述转换为高质量的Stable Diffusion提示词，遵循以下规则：

    1. 使用英文输出
    2. 包含详细的主体描述、环境背景、艺术风格和画质增强词
    3. 添加合适的艺术家或风格参考（如：by Studio Ghibli, Van Gogh style）
    4. 包含画质关键词：masterpiece, best quality, ultra-detailed, 8k
    5. 添加负面提示词：low quality, blurry, deformed, text, watermark
    6. 保持提示词长度在15-30个单词之间

    输出格式:
    [优化后的提示词]
    [负面提示词]
    """

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": natural_language}
        ],
        "temperature": 0.7,
        "max_tokens": 200
    }

    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        full_response = result["choices"][0]["message"]["content"].strip()

        # 解析优化后的提示词和负面提示词
        parts = full_response.split("\n")
        prompt = parts[1].strip()
        negative_prompt = "low quality, blurry, deformed, text, watermark"

        print(f"优化提示词: {prompt}")
        print(f"负面提示词: {negative_prompt}")

        return prompt, negative_prompt
    else:
        raise Exception(f"DeepSeek API 请求失败: {response.status_code}, {response.text}")


def generate_image(prompt, negative_prompt):
    """使用优化后的提示词生成图像"""
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": 25,
        "width": 768,
        "height": 512,
        "sampler_name": "DPM++ 2M Karras",
        "cfg_scale": 7.5,
        "seed": -1,
    }

    response = requests.post(SD_API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        image_data = result['images'][0]

        # 保存图像
        image = Image.open(BytesIO(base64.b64decode(image_data.split(",", 1)[0])))
        filename = f"output\generated_image_{hash(prompt)}.png"
        image.save(filename)

        print(f"图像已保存至: {filename}")
        return filename
    else:
        raise Exception(f"Stable Diffusion API 请求失败: {response.status_code}, {response.text}")


def main():
    """主函数：获取用户输入并生成图像"""
    print("=" * 50)
    print("Stable Diffusion 图像生成器 (DeepSeek 提示词优化)")
    print("=" * 50)

    # 获取用户输入
    user_input = input("请输入图像描述 (自然语言): ")

    try:
        # 优化提示词
        prompt, negative_prompt = generate_prompt(user_input)

        # 生成图像
        print("\n正在生成图像...")
        image_path = generate_image(prompt, negative_prompt)

        print("\n生成完成! 请查看图像文件:", image_path)

    except Exception as e:
        print(f"错误发生: {str(e)}")


if __name__ == "__main__":
    main()