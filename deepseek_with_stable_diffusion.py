import requests
import base64
import os
from PIL import Image
from io import BytesIO
import tempfile
# DeepSeek API 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-91a54db58fc64524b68957de65517a57")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Stable Diffusion API 配置
SD_API_URL = "http://127.0.0.1:7860/sdapi/v1/txt2img"

# LoRA 模型配置
LORA_MODELS = {
    "ghibli": {
        "name": "ghibli_style_offset",
        "weight": 0.8,
        "trigger": "ghibli_style_offset"
    },
    "MUTOU": {
        "name": "MUTOU",
        "weight": 1.2,
        "trigger": "ruoyemu"
    },
}

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


def generate_prompt(natural_language, lora_key=None):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

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

    if lora_key and lora_key in LORA_MODELS:
        lora = LORA_MODELS[lora_key]
        system_prompt += f"\n5. 使用 {lora['name']} LoRA 模型（权重 {lora['weight']}）"
        if lora.get('trigger'):
            system_prompt += f"，包含触发词 '{lora['trigger']}'"

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

        # 解析优化后的提示词
        prompt_lines = full_response.split("\n")
        prompt = prompt_lines[0].strip()
        if prompt.startswith('[') and prompt.endswith(']'):
            prompt = prompt[1:-1].strip()

        # 添加 LoRA 标记
        if lora_key and lora_key in LORA_MODELS:
            lora = LORA_MODELS[lora_key]
            prompt += f" <lora:{lora['name']}:{lora['weight']}>"
            if lora.get('trigger'):
                prompt = f"{lora['trigger']}, " + prompt

        # 获取负面提示词
        negative_prompt = "low quality, blurry, deformed, text, watermark"

        print(f"优化提示词: {prompt}")
        print(f"负面提示词: {negative_prompt}")

        return prompt, negative_prompt
    else:
        raise Exception(f"DeepSeek API 请求失败: {response.status_code}, {response.text}")


def generate_image(prompt, negative_prompt):
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
        #image.show()
        # 使用subprocess打开图像，避免编码问题
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_filename = tmp_file.name
            image.save(temp_filename)

        # 使用系统命令打开图像文件
        os.startfile(temp_filename)

        filename = f"output\generated_image_{hash(prompt)}.png"
        image.save(filename)

        print(f"图像已保存至: {filename}")
        return filename
    else:
        raise Exception(f"Stable Diffusion API 请求失败: {response.status_code}, {response.text}")


def list_available_loras():
    """列出可用的 LoRA 模型"""
    print("\n可用 LoRA 模型:")
    for i, key in enumerate(LORA_MODELS.keys(), 1):
        lora = LORA_MODELS[key]
        print(f"{i}. {key} ({lora['name']}) - 权重: {lora['weight']}")
    print("0. 不使用 LoRA\n")


def main():
    print("=" * 50)
    print("Stable Diffusion 图像生成器 (支持 LoRA 模型)")
    print("=" * 50)

    # 获取用户输入
    user_input = input("请输入图像描述 (自然语言): ")

    # 选择 LoRA 模型
    list_available_loras()
    lora_choice = input("请选择要使用的 LoRA 模型 (输入编号): ")

    lora_key = None
    if lora_choice.isdigit():
        choice_index = int(lora_choice)
        if 1 <= choice_index <= len(LORA_MODELS):
            lora_key = list(LORA_MODELS.keys())[choice_index - 1]

    try:
        # 优化提示词（包含 LoRA）
        prompt, negative_prompt = generate_prompt(user_input, lora_key)

        # 生成图像
        print("\n正在生成图像...")
        image_path = generate_image(prompt, negative_prompt)

        print("\n生成完成! 请查看图像文件:", image_path)

    except Exception as e:
        print(f"错误发生: {str(e)}")


if __name__ == "__main__":
    main()
