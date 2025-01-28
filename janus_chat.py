# 导入基础依赖
import os
import sys
import json
import torch
import time
import numpy as np
import cv2
from PIL import Image
import io
import folder_paths
import comfy.model_management
from typing import List, Optional, Dict, Any, Union

# 从transformers导入必要组件
from transformers import AutoModelForCausalLM

# 导入janus组件
from janus.models import (
    MultiModalityCausalLM,
    VLChatProcessor,
)

# 获取设备
device = comfy.model_management.get_torch_device()

def resize_with_aspect_ratio(img, target_size, target_dim='width', interpolation=cv2.INTER_CUBIC):
    """等比例缩放图片"""
    h, w = img.shape[:2]
    if target_dim == 'width':
        aspect = h / w
        new_w = target_size
        new_h = int(aspect * new_w)
    else:
        aspect = w / h
        new_h = target_size
        new_w = int(aspect * new_h)
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)

def calculate_aspect_ratio(width, height):
    """计算宽高比"""
    return width / height

def draw_border(image, color=(0, 255, 0), thickness=2):
    """在图片周围绘制边框"""
    h, w = image.shape[:2]
    cv2.rectangle(image, (0, 0), (w-1, h-1), color, thickness)
    return image

def combine_images(first_image, second_image=None, target_size=1024, border_thickness=2):
    """组合图片处理函数"""
    # 处理单图情况
    if second_image is None:
        first_image = (first_image[0].detach().cpu().numpy() * 255).astype(np.uint8)
        first_image = draw_border(first_image.copy(), thickness=border_thickness)
        # 调整尺寸
        h, w = first_image.shape[:2]
        ratio = w / h
        if ratio > 1:
            new_width = target_size
            new_height = int(target_size / ratio)
        else:
            new_height = target_size
            new_width = int(target_size * ratio)
        first_image = cv2.resize(first_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        return first_image, "single"

    # 处理双图情况
    first_image = (first_image[0].detach().cpu().numpy() * 255).astype(np.uint8)
    second_image = (second_image[0].detach().cpu().numpy() * 255).astype(np.uint8)

    # 计算拼接方式
    h1, w1 = first_image.shape[:2]
    h2, w2 = second_image.shape[:2]
    horizontal_ratio = calculate_aspect_ratio(w1 + w2, max(h1, h2))
    vertical_ratio = calculate_aspect_ratio(max(w1, w2), h1 + h2)
    use_horizontal = abs(horizontal_ratio - 1.33) < abs(vertical_ratio - 1.33)

    if use_horizontal:
        # 水平拼接
        target_height = min(h1, h2)
        first_image = resize_with_aspect_ratio(first_image, target_height, 'height')
        second_image = resize_with_aspect_ratio(second_image, target_height, 'height')
        h1, w1 = first_image.shape[:2]
        h2, w2 = second_image.shape[:2]
        first_image = draw_border(first_image.copy(), thickness=border_thickness)
        second_image = draw_border(second_image.copy(), thickness=border_thickness)
        combined_width = w1 + w2
        combined_height = max(h1, h2)
        combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        combined_image[:h1, :w1] = first_image
        combined_image[:h2, w1:w1+w2] = second_image
        layout = "horizontal"
    else:
        # 垂直拼接
        target_width = min(w1, w2)
        first_image = resize_with_aspect_ratio(first_image, target_width, 'width')
        second_image = resize_with_aspect_ratio(second_image, target_width, 'width')
        h1, w1 = first_image.shape[:2]
        h2, w2 = second_image.shape[:2]
        first_image = draw_border(first_image.copy(), thickness=border_thickness)
        second_image = draw_border(second_image.copy(), thickness=border_thickness)
        combined_width = max(w1, w2)
        combined_height = h1 + h2
        combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        combined_image[:h1, :w1] = first_image
        combined_image[h1:h1+h2, :w2] = second_image
        layout = "vertical"

    # 调整最终尺寸
    final_ratio = calculate_aspect_ratio(combined_width, combined_height)
    if final_ratio > 1:
        new_width = target_size
        new_height = int(target_size / final_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * final_ratio)
    combined_image = cv2.resize(combined_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    return combined_image, layout

class JanusChatAnalyzer:
    """Janus多轮对话分析器"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "janus_model": ("JANUS_MODEL",),
                "image_a": ("IMAGE",),
                "question": ("STRING", {
                    "multiline": True,
                    "default": "请描述这张图片。",
                    "placeholder": "请输入您的问题"
                }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 2048}),
                "target_size": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 64
                }),
                "border_thickness": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
                "clear_history": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image_b": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)  # (response, updated_history)
    RETURN_NAMES = ("response", "chat_history")
    FUNCTION = "chat_analyze"
    CATEGORY = "Yc_JanusPro"

    def __init__(self):
        """初始化对话历史"""
        self.conversation_history = []

    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        """格式化对话历史，使其更易读"""
        try:
            formatted_history = []
            for entry in history:
                # 格式化每条对话记录
                timestamp = entry.get("timestamp", "")
                role = entry.get("role", "").replace("<|", "").replace("|>", "")
                content = entry.get("content", "")
                
                # 构建格式化的对话记录
                formatted_entry = f"[{timestamp}] {role}:\n{content}\n"
                if "images" in entry:
                    formatted_entry += "[包含图片]\n"
                formatted_entry += "-" * 50 + "\n"  # 分隔线
                
                formatted_history.append(formatted_entry)
            
            # 合并所有对话记录
            return "\n".join(formatted_history)
        except Exception as e:
            print(f"格式化历史记录时出错: {str(e)}")
            return str(history)

    def _prepare_images(self, image_a: torch.Tensor, image_b: Optional[torch.Tensor] = None, 
                       target_size: int = 1024, border_thickness: int = 2) -> List[Image.Image]:
        """准备图片"""
        # 使用新的图片组合功能
        combined_image, layout = combine_images(image_a, image_b, target_size, border_thickness)
        
        # 转换为PIL图片
        if isinstance(combined_image, np.ndarray):
            combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
            combined_image = Image.fromarray(combined_image)
        
        return [combined_image], layout

    def _build_prompt(self, question: str, images: List[Image.Image], layout: str = "single") -> str:
        """构建结构化的提示词"""
        if layout == "single":
            return question
        else:
            first_position = "左" if layout == "horizontal" else "上"
            second_position = "右" if layout == "horizontal" else "下"
            
            return f"""{first_position}边是第一张图片，{second_position}边是第二张图片。

{question}"""

    def _clear_history(self):
        """清除对话历史"""
        self.conversation_history = []

    def chat_analyze(self, janus_model, image_a, question, 
                    seed=42, temperature=0.1, top_p=0.95, max_new_tokens=512,
                    target_size=1024, border_thickness=2, clear_history=False,
                    image_b=None):
        try:
            # 检查是否需要清除历史
            if clear_history:
                self._clear_history()
                return ("已清除对话历史。", "")

            # 处理模型和处理器
            if isinstance(janus_model, dict) and "error" in janus_model:
                return (f"错误: {janus_model['error']}", "")
            
            if isinstance(janus_model, dict):
                model = janus_model.get("model")
                processor = janus_model.get("processor")
            else:
                model = janus_model
                processor = getattr(janus_model, "processor", None)
            
            if model is None or processor is None:
                return ("错误: 模型或处理器未正确加载", "")

            # 设置随机种子
            if seed != -1:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
            
            # 准备图片
            images, layout = self._prepare_images(image_a, image_b, target_size, border_thickness)
            
            # 构建提示词
            prompt = self._build_prompt(question, images, layout)
            
            # 准备对话上下文
            conversation_context = []
            
            # 添加历史对话（最近3轮）
            if len(self.conversation_history) > 0:
                for hist in self.conversation_history[-3:]:
                    if hist.get("images"):
                        # 处理带图片的历史记录
                        hist_copy = hist.copy()
                        hist_copy["images"] = hist["images"][:1]  # 每条记录只保留一张图片
                        conversation_context.append(hist_copy)
                    else:
                        conversation_context.append(hist)

            # 构建当前对话
            current_conversation = {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{prompt}",
                "images": images,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            conversation_context.append(current_conversation)

            # 添加Assistant角色
            conversation_context.append({"role": "<|Assistant|>", "content": ""})
            
            # 准备输入
            try:
                inputs = processor(
                    conversations=conversation_context,
                    images=images,
                    force_batchify=True
                ).to(device)
            except RuntimeError as e:
                print(f"处理输入时出错: {str(e)}")
                # 尝试不使用force_batchify
                inputs = processor(
                    conversations=conversation_context,
                    images=images
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
            
            # 生成回答
            with torch.inference_mode():
                outputs = model.language_model.generate(**generation_config)
            
            # 解码输出
            response = processor.tokenizer.decode(
                outputs[0].cpu().tolist(), 
                skip_special_tokens=True
            ).strip()
            
            # 更新对话历史
            assistant_response = {
                "role": "<|Assistant|>",
                "content": response,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 更新类的对话历史
            self.conversation_history.append(current_conversation)
            self.conversation_history.append(assistant_response)
            
            # 保持最近10轮对话
            if len(self.conversation_history) > 20:  # 10轮对话=20条记录
                self.conversation_history = self.conversation_history[-20:]
            
            # 格式化并返回结果
            formatted_history = self._format_history(self.conversation_history)
            return (response, formatted_history)

        except Exception as e:
            error_msg = f"处理过程中出错: {str(e)}"
            print(error_msg)
            return (error_msg, "")

NODE_CLASS_MAPPINGS = {
    "JanusChatAnalyzer": JanusChatAnalyzer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JanusChatAnalyzer": "Janus Chat"
} 