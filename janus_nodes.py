# 导入基础依赖
import os
import sys
import torch
import numpy as np  # 添加numpy导入
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
                
                print(f"正在加载Janus-Pro模型，路径：{model_folder}")
                
                # 加载处理器
                try:
                    self.processor = VLChatProcessor.from_pretrained(model_folder)
                    print("处理器加载成功")
                    print(f"处理器类型: {type(self.processor)}")
                except Exception as e:
                    print(f"处理器加载失败: {str(e)}")
                    raise
                
                # 加载模型
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_folder,
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16,
                        device_map=device,
                        use_safetensors=False,
                        low_cpu_mem_usage=True,
                    )
                    print("模型加载成功")
                    print(f"模型类型: {type(self.model)}")
                except Exception as e:
                    print(f"模型加载失败: {str(e)}")
                    raise
                
                # 移动模型到设备并设置为评估模式
                try:
                    self.model = self.model.to(device).eval()
                    print(f"模型已移动到设备: {device}")
                except Exception as e:
                    print(f"模型迁移到设备失败: {str(e)}")
                    raise
                
                # 验证模型和处理器的关键属性和方法
                try:
                    # 验证处理器
                    assert hasattr(self.processor, 'tokenizer'), "处理器缺少tokenizer"
                    assert hasattr(self.processor, 'pad_id'), "处理器缺少pad_id"
                    
                    # 验证模型
                    assert hasattr(self.model, 'gen_vision_model'), "模型缺少gen_vision_model"
                    assert hasattr(self.model, 'gen_head'), "模型缺少gen_head"
                    assert hasattr(self.model, 'prepare_gen_img_embeds'), "模型缺少prepare_gen_img_embeds"
                    
                    print("模型和处理器验证通过")
                except AssertionError as e:
                    print(f"模型验证失败: {str(e)}")
                    raise
                
                self.loaded_model_path = model_path
                print("Janus-Pro模型加载完成")
                
            return ({"model": self.model, "processor": self.processor},)
            
        except Exception as e:
            error_msg = f"模型加载错误: {str(e)}"
            print(error_msg)
            traceback.print_exc()
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

class JanusImageGenerator:
    """Janus Pro 文本生成图像节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "janus_model": ("JANUS_MODEL",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A stunning princess",
                    "placeholder": "请输入图片生成提示词"
                }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "cfg_weight": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate_images"
    CATEGORY = "Yc_JanusPro"

    def prepare_prompt(self, prompt: str, processor) -> str:
        """准备生成提示词"""
        # 直接使用原始提示词
        return prompt

    @torch.inference_mode()
    def generate_images(
        self,
        janus_model,
        prompt: str,
        seed: int = 42,
        control_after_generate: str = "fixed",
        batch_size: int = 1,
        cfg_weight: float = 5.0,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ):
        try:
            print(f"\n=== Janus Text2Image ===")
            print(f"提示词: {prompt}")
            print(f"种子: {seed}")
            
            # 参数设置
            image_size = 384
            patch_size = 16
            parallel_size = batch_size
            
            # 验证模型
            if isinstance(janus_model, dict) and "error" in janus_model:
                print(f"错误: 模型加载失败 - {janus_model['error']}")
                return (torch.zeros((batch_size, image_size, image_size, 3), dtype=torch.float32),)
            
            # 获取模型和处理器
            if isinstance(janus_model, dict):
                model = janus_model.get("model")
                processor = janus_model.get("processor")
            else:
                model = janus_model
                processor = getattr(janus_model, "processor", None)
            
            if model is None or processor is None:
                print("错误: 模型或处理器未正确加载")
                return (torch.zeros((batch_size, image_size, image_size, 3), dtype=torch.float32),)

            # 设置随机种子
            if seed != -1:
                # 根据控制模式处理种子
                if control_after_generate == "randomize":
                    seed = torch.randint(0, 0xffffffffffffffff, (1,)).item()
                elif control_after_generate == "increment":
                    seed += 1
                elif control_after_generate == "decrement":
                    seed -= 1
                
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                print(f"最终种子: {seed}")

            # 准备对话格式
            conversation = [
                {
                    "role": "<|User|>",
                    "content": prompt,
                },
                {"role": "<|Assistant|>", "content": ""},
            ]

            # 应用模板
            prompt = processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=processor.sft_format,
                system_prompt="",
            )
            prompt = prompt + processor.image_start_tag
            
            # Token生成
            image_token_num = 576
            input_ids = processor.tokenizer.encode(prompt)
            input_ids = torch.LongTensor(input_ids)

            # 准备输入
            tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
            for i in range(parallel_size*2):
                tokens[i, :] = input_ids
                if i % 2 != 0:
                    tokens[i, 1:-1] = processor.pad_id

            # 获取输入嵌入
            inputs_embeds = model.language_model.get_input_embeddings()(tokens)

            # 初始化生成tokens
            generated_tokens = torch.zeros((parallel_size, image_token_num), dtype=torch.int).cuda()

            # 自回归生成过程
            outputs = None
            print("开始生成...")
            for i in range(image_token_num):
                if i % 100 == 0:  # 每100步打印一次进度
                    print(f"生成进度: {i}/{image_token_num}")
                
                outputs = model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    past_key_values=outputs.past_key_values if i != 0 else None
                )
                hidden_states = outputs.last_hidden_state
                
                logits = model.gen_head(hidden_states[:, -1, :])
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]
                
                logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
                
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    if top_p < 1.0:
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        probs = probs.masked_fill(indices_to_remove, 0.0)
                        probs = probs / probs.sum(dim=-1, keepdim=True)
                else:
                    probs = torch.zeros_like(logits).scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)

                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens[:, i] = next_token.squeeze(dim=-1)

                next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
                img_embeds = model.prepare_gen_img_embeds(next_token)
                inputs_embeds = img_embeds.unsqueeze(1)

            print("解码图像...")
            # 解码生成的图像
            dec = model.gen_vision_model.decode_code(
                generated_tokens.to(dtype=torch.int),
                shape=[parallel_size, 8, image_size//patch_size, image_size//patch_size]
            )
            
            # 转换为numpy并处理
            dec = dec.to(torch.float32).cpu().numpy()
            dec = dec.transpose(0, 2, 3, 1)
            
            # 值域处理
            dec = np.clip((dec + 1) / 2 * 255, 0, 255)
            
            # 创建最终图像数组
            visual_img = np.zeros((parallel_size, image_size, image_size, 3), dtype=np.uint8)
            visual_img[:, :, :] = dec
            
            # 转换所有图片为tensor格式
            images_list = []
            for i in range(parallel_size):
                image = visual_img[i]
                image_float = image.astype(np.float32) / 255.0
                images_list.append(image_float)
            
            # 堆叠所有图片
            images_tensor = torch.from_numpy(np.stack(images_list))
            
            print(f"生成完成，共 {parallel_size} 张图片")
            return (images_tensor,)

        except Exception as e:
            print("\n=== 发生错误 ===")
            import traceback
            error_msg = f"生成过程中出现错误:\n{str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return (torch.zeros((1, image_size, image_size, 3), dtype=torch.float32),)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

# 节点映射
NODE_CLASS_MAPPINGS = {
    "JanusProLoader": JanusProLoader,
    "ImageAnalyzer": ImageAnalyzer,
    "JanusImageGenerator": JanusImageGenerator
}

# 显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "JanusProLoader": "Load Janus-Pro Model",
    "ImageAnalyzer": "Analyze Image with Janus",
    "JanusImageGenerator": "Janus Text2Image"
}

