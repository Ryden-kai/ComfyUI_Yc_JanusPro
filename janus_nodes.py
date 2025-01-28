# 导入基础依赖
import os
import sys
import torch
from PIL import Image
import folder_paths
import comfy.model_management
from typing import List, Optional

# 从transformers导入必要组件
from transformers import AutoModelForCausalLM

# 如果janus已经作为依赖包安装在虚拟环境中，直接导入所需组件
from janus.models import (
    MultiModalityCausalLM,  # 多模态因果语言模型
    VLChatProcessor,        # 视觉语言处理器
)

# 获取设备
device = comfy.model_management.get_torch_device()

# 设置模型路径
current_path = os.path.dirname(os.path.realpath(__file__))
models_path = folder_paths.models_dir
janus_model_path = os.path.join(models_path, "Janus-Pro")

# 确保模型目录存在
if not os.path.exists(janus_model_path):
    os.makedirs(janus_model_path, exist_ok=True)

# 注册模型路径
folder_paths.add_model_folder_path("janus_model", janus_model_path)

class JanusProLoader:
    """Janus-Pro模型加载器"""
    
    # 添加必要的类属性
    RETURN_TYPES = ("JANUS_MODEL",)
    RETURN_NAMES = ("janus_model",)
    FUNCTION = "load_model"
    CATEGORY = "Yc_JanusPro"
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.loaded_model_path = None

    @classmethod
    def INPUT_TYPES(s):
        """获取可用的模型列表"""
        model_paths = []
        if os.path.exists(janus_model_path):
            # 检查目录下的所有文件夹
            for model_dir in os.listdir(janus_model_path):
                model_dir_path = os.path.join(janus_model_path, model_dir)
                # 检查是否是目录且包含必要的模型文件
                if os.path.isdir(model_dir_path) and s._is_valid_model_dir(model_dir_path):
                    model_paths.append(model_dir)
        
        return {
            "required": {
                "model_path": (model_paths if model_paths else ["请将模型文件放置在models/Janus-Pro目录下"],),
            },
        }

    @staticmethod
    def _is_valid_model_dir(model_dir: str) -> bool:
        """检查模型目录是否包含必要的文件"""
        # 检查是否存在以下任一文件组合
        file_combinations = [
            # 完整模型文件
            ["config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json"],
            # 分片模型文件
            ["config.json", "tokenizer.json", "tokenizer_config.json"],
        ]
        
        for combination in file_combinations:
            # 检查是否存在分片模型文件
            if all(os.path.exists(os.path.join(model_dir, file)) for file in combination):
                # 如果是分片模型，检查是否存在任何.bin分片文件
                if "pytorch_model.bin" not in combination:
                    bin_files = [f for f in os.listdir(model_dir) if f.startswith("pytorch_model-") and f.endswith(".bin")]
                    if bin_files:  # 如果找到任何分片文件
                        return True
                else:
                    return True
        return False

    def load_model(self, model_path):
        """加载模型"""
        try:
            if self.loaded_model_path != model_path or self.model is None or self.processor is None:
                model_folder = os.path.join(janus_model_path, model_path)
                
                if not self._is_valid_model_dir(model_folder):
                    raise ValueError(f"模型目录 {model_path} 缺少必要的文件")
                
                print(f"正在加载Janus-Pro模型，路径：{model_folder}...")
                
                self.processor = VLChatProcessor.from_pretrained(model_folder)
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_folder,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                    # 添加分片模型支持
                    use_safetensors=False,  # 如果模型不是safetensors格式
                    low_cpu_mem_usage=True,  # 低内存加载
                )
                self.model = self.model.to(device).eval()
                
                self.loaded_model_path = model_path
                print("Janus-Pro模型加载成功")
                
            return ({"model": self.model, "processor": self.processor},)
        except Exception as e:
            error_msg = f"模型加载错误: {str(e)}"
            print(error_msg)
            return ({"error": error_msg, "model": None, "processor": None},)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """检测参数变化"""
        return float("nan")

class ImageAnalyzer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "janus_model": ("JANUS_MODEL",),
                "image": ("IMAGE",),
                "question": ("STRING", {
                    "multiline": True,
                    "default": "请描述这张图片。",
                    "placeholder": "请输入您想问的问题"
                }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 2048}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "analyze_image"
    CATEGORY = "Yc_JanusPro"

    def analyze_image(self, janus_model, image, question="请描述这张图片。", 
                     seed=42, temperature=0.1, top_p=0.95, max_new_tokens=512):
        try:
            # 修改错误检查逻辑
            if isinstance(janus_model, dict) and "error" in janus_model:
                return (f"错误: {janus_model['error']}",)
            
            # 获取模型和处理器
            if isinstance(janus_model, dict):
                model = janus_model.get("model")
                processor = janus_model.get("processor")
            else:
                # 如果直接传入的是模型对象
                model = janus_model
                processor = getattr(janus_model, "processor", None)
            
            if model is None or processor is None:
                return ("错误: 模型或处理器未正确加载",)

            # 设置随机种子
            if seed != -1:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
            
            # 转换图像格式
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
                image = (image * 255).astype('uint8')
                if image.shape[0] == 1:
                    image = image[0]
                image = Image.fromarray(image).convert('RGB')
            
            # 准备对话
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{question}",
                    "images": [image],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            
            # 准备输入
            inputs = processor(
                conversations=conversation,
                images=[image],
                force_batchify=True
            ).to(device)
            
            # 生成回答
            inputs_embeds = model.prepare_inputs_embeds(**inputs)
            
            # 配置生成参数
            generation_config = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": inputs.attention_mask,
                "pad_token_id": processor.tokenizer.eos_token_id,
                "bos_token_id": processor.tokenizer.bos_token_id,
                "eos_token_id": processor.tokenizer.eos_token_id,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": temperature > 0,
                "use_cache": True,
            }
            
            outputs = model.language_model.generate(**generation_config)
            
            # 解码输出
            answer = processor.tokenizer.decode(
                outputs[0].cpu().tolist(), 
                skip_special_tokens=True
            )
            
            # 清理和格式化输出
            answer = answer.replace(question, "").strip()
            
            return (answer,)
            
        except Exception as e:
            import traceback
            error_msg = f"分析过程中出现错误:\n{str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return (error_msg,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """检测参数变化"""
        return float("nan")

# 节点映射
NODE_CLASS_MAPPINGS = {
    "JanusProLoader": JanusProLoader,
    "ImageAnalyzer": ImageAnalyzer
}

# 显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "JanusProLoader": "Load Janus-Pro Model",
    "ImageAnalyzer": "Analyze Image with Janus"
} 