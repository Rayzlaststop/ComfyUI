import torch

import os
import sys
import json
import hashlib
import traceback
import math
import time
import random

from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
import numpy as np
import safetensors.torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))


import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils

import comfy.clip_vision

import comfy.model_management
from comfy.cli_args import args

import importlib

import folder_paths
import latent_preview

def before_node_execution():
    comfy.model_management.throw_exception_if_processing_interrupted()

def interrupt_processing(value=True):
    comfy.model_management.interrupt_current_processing(value)

MAX_RESOLUTION=8192


#采样器
def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    device = comfy.model_management.get_torch_device()
    latent_image = latent["samples"]

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    previewer = latent_preview.get_previewer(device, model.model.latent_format)

    pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps):
        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)

    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return (out, )

# KSampler
class 采样器_Zho:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"模型": ("MODEL",),
                    "种子": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "步数": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "CFG值": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "采样器": (comfy.samplers.KSampler.SAMPLERS, ),
                    "调度器": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "正向提示词": ("CONDITIONING", ),
                    "负向提示词": ("CONDITIONING", ),
                    "潜空间图像": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     }
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("潜空间图像",)
    FUNCTION = "sample"

    CATEGORY = "Zho汉化模块组/采样器"

    def sample(self, 模型, 种子, 步数, CFG值, 采样器, 调度器, 正向提示词, 负向提示词, 潜空间图像, denoise=1.0):
        return common_ksampler(模型, 种子, 步数, CFG值, 采样器, 调度器, 正向提示词, 负向提示词, 潜空间图像, denoise=denoise)

# KSamplerAdvanced
class 高级采样器_Zho:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"模型": ("MODEL",),
                    "增加噪点": (["开启", "关闭"], ),
                    "噪点种子": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "步数": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "CFG值": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "采样器": (comfy.samplers.KSampler.SAMPLERS, ),
                    "调度器": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "正向提示词": ("CONDITIONING", ),
                    "负向提示词": ("CONDITIONING", ),
                    "潜空间图像": ("LATENT", ),
                    "起始步数": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "结束步数": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "返回剩余噪点": (["关闭", "开启"], ),
                     }
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("潜空间图像",)
    FUNCTION = "sample"

    CATEGORY = "Zho汉化模块组/采样器"

    def sample(self, 模型, 增加噪点, 噪点种子, 步数, CFG值, 采样器, 调度器, 正向提示词, 负向提示词, 潜空间图像, 起始步数, 结束步数, 返回剩余噪点, denoise=1.0):
        force_full_denoise = True
        if 返回剩余噪点 == "开启":
            force_full_denoise = False
        disable_noise = False
        if 增加噪点 == "关闭":
            disable_noise = True
        return common_ksampler(模型, 噪点种子, 步数, CFG值, 采样器, 调度器, 正向提示词, 负向提示词, 潜空间图像, denoise=denoise, disable_noise=disable_noise, start_step=起始步数, last_step=结束步数, force_full_denoise=force_full_denoise)

#---------------------------------
#加载器
# CheckpointLoaderSimple
class 主模型加载器_Zho:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "主模型名称": (folder_paths.get_filename_list("checkpoints"), ),
                             }}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("主模型", "CLIP模型", "VAE模型")
    FUNCTION = "load_checkpoint"

    CATEGORY = "Zho汉化模块组/加载器"

    def load_checkpoint(self, 主模型名称, output_vae=True, output_clip=True):
        ckpt_path = folder_paths.get_full_path("checkpoints", 主模型名称)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=output_vae, output_clip=output_clip, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out

# VAELoader
class VAE加载器_Zho:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "VAE模型名称": (folder_paths.get_filename_list("vae"), )}}
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("VAE模型",)
    FUNCTION = "load_vae"

    CATEGORY = "Zho汉化模块组/加载器"

    # TODO: scale factor?
    def load_vae(self, VAE模型名称):
        vae_path = folder_paths.get_full_path("vae", VAE模型名称)
        vae = comfy.sd.VAE(ckpt_path=vae_path)
        return (vae,)

# LoraLoader
class Lora加载器_Zho:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "lora_name": (folder_paths.get_filename_list("loras"), ),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("主模型", "CLIP模型")
    FUNCTION = "load_lora"

    CATEGORY = "Zho汉化模块组/加载器"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)

# ControlNetLoader
class ControlNet加载器_Zho:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "CtrlNet模型": (folder_paths.get_filename_list("controlnet"), )}}

    RETURN_TYPES = ("CONTROL_NET",)
    RETURN_NAMES = ("CtrlNet",)
    FUNCTION = "load_controlnet"

    CATEGORY = "Zho汉化模块组/加载器"

    def load_controlnet(self, CtrlNet模型):
        controlnet_path = folder_paths.get_full_path("controlnet", CtrlNet模型)
        controlnet = comfy.sd.load_controlnet(controlnet_path)
        return (controlnet,)

# GLIGENLoader
class GLIGEN加载器_Zho:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "GLIGEN模型": (folder_paths.get_filename_list("gligen"), )}}

    RETURN_TYPES = ("GLIGEN",)
    RETURN_NAMES = ("GLIGEN模型",)
    FUNCTION = "load_gligen"

    CATEGORY = "Zho汉化模块组/加载器"

    def load_gligen(self, GLIGEN模型):
        gligen_path = folder_paths.get_full_path("gligen", GLIGEN模型)
        gligen = comfy.sd.load_gligen(gligen_path)
        return (gligen,)

#---------------------------------
#条件condition
# CLIPTextEncode
class 提示词_Zho:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"文本": ("STRING", {"multiline": True}), "CLIP模型": ("CLIP", )}}
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("条件",)
    FUNCTION = "encode"

    CATEGORY = "Zho汉化模块组/条件"

    def encode(self, CLIP模型, 文本):
        tokens = CLIP模型.tokenize(文本)
        cond, pooled = CLIP模型.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], )

# CLIP跳过层
class CLIP跳过层_Zho:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "CLIP模型": ("CLIP", ),
                              "CLIP跳过层": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
                              }}
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("CLIP模型",)
    FUNCTION = "set_last_layer"

    CATEGORY = "Zho汉化模块组/条件"

    def set_last_layer(self, CLIP模型, CLIP跳过层):
        clip = CLIP模型.clone()
        clip.clip_layer(CLIP跳过层)
        return (clip,)

# GLIGEN
class GLIGEN区域设定_Zho:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"条件去往": ("CONDITIONING", ),
                              "CLIP模型": ("CLIP", ),
                              "GLIGEN文字区域设定": ("GLIGEN", ),
                              "文本": ("STRING", {"multiline": True}),
                              "宽度": ("INT", {"default": 64, "min": 8, "max": MAX_RESOLUTION, "step": 8}),
                              "高度": ("INT", {"default": 64, "min": 8, "max": MAX_RESOLUTION, "step": 8}),
                              "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                              "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("条件",)
    FUNCTION = "append"

    CATEGORY = "Zho汉化模块组/条件"

    def append(self, 条件去往, CLIP模型, GLIGEN文字区域设定, 文本, 宽度, 高度, x, y):
        c = []
        cond, cond_pooled = CLIP模型.encode_from_tokens(CLIP模型.tokenize(文本), return_pooled=True)
        for t in 条件去往:
            n = [t[0], t[1].copy()]
            position_params = [(cond_pooled, 高度 // 8, 宽度 // 8, y // 8, x // 8)]
            prev = []
            if "gligen" in n[1]:
                prev = n[1]['gligen'][2]

            n[1]['gligen'] = ("position", GLIGEN文字区域设定, prev + position_params)
            c.append(n)
        return (c, )

# CN
class ControlNet_Zho:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"条件": ("CONDITIONING", ),
                             "CrtlNet": ("CONTROL_NET", ),
                             "图像": ("IMAGE", ),
                             "强度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("条件",)
    FUNCTION = "apply_controlnet"

    CATEGORY = "Zho汉化模块组/条件"

    def apply_controlnet(self, 条件, CrtlNet, 图像, 强度):
        if 强度 == 0:
            return (条件, )

        c = []
        control_hint = 图像.movedim(-1, 1)
        for t in 条件:
            n = [t[0], t[1].copy()]
            c_net = CrtlNet.copy().set_cond_hint(control_hint, 强度)
            if 'control' in t[1]:
                c_net.set_previous_controlnet(t[1]['control'])
            n[1]['control'] = c_net
            n[1]['control_apply_to_uncond'] = True
            c.append(n)
        return (c, )

#---------------------------------
#条件latent
# 空白潜空间图像
class 初始潜空间_Zho:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "宽度": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                              "高度": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                              "批次数": ("INT", {"default": 1, "min": 1, "max": 64})}}
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("潜空间",)
    FUNCTION = "generate"

    CATEGORY = "Zho汉化模块组/潜空间"

    def generate(self, 宽度, 高度, 批次数=1):
        latent = torch.zeros([批次数, 4, 高度 // 8, 宽度 // 8])
        return ({"samples":latent}, )

# VAE解码
class VAE解码器_Zho:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT", ), "VAE模型": ("VAE", )}}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "decode"

    CATEGORY = "Zho汉化模块组/潜空间"

    def decode(self, VAE模型, samples):
        return (VAE模型.decode(samples["samples"]), )

# VAE编码
class VAE编码器_Zho:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "像素": ("IMAGE", ), "VAE模型": ("VAE", )}}
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("潜空间",)
    FUNCTION = "encode"

    CATEGORY = "Zho汉化模块组/潜空间"

    @staticmethod
    def vae_encode_crop_pixels(像素):
        x = (像素.shape[1] // 8) * 8
        y = (像素.shape[2] // 8) * 8
        if 像素.shape[1] != x or 像素.shape[2] != y:
            x_offset = (像素.shape[1] % 8) // 2
            y_offset = (像素.shape[2] % 8) // 2
            像素 = 像素[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return 像素

    def encode(self, VAE模型, 像素):
        像素 = self.vae_encode_crop_pixels(像素)
        t = VAE模型.encode(像素[:,:,:,:3])
        return ({"samples":t}, )

# 提示词VAE编码 inpaint
class VAE编码器_重绘_Zho:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "像素": ("IMAGE", ), "VAE模型": ("VAE", ), "蒙版": ("MASK", ), "扩大蒙版": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),}}
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("潜空间",)
    FUNCTION = "encode"

    CATEGORY = "Zho汉化模块组/潜空间"

    def encode(self, VAE模型, 像素, 蒙版, 扩大蒙版=6):
        x = (像素.shape[1] // 8) * 8
        y = (像素.shape[2] // 8) * 8
        蒙版 = torch.nn.functional.interpolate(蒙版.reshape((-1, 1, 蒙版.shape[-2], 蒙版.shape[-1])), size=(像素.shape[1], 像素.shape[2]), mode="bilinear")

        像素 = 像素.clone()
        if 像素.shape[1] != x or 像素.shape[2] != y:
            x_offset = (像素.shape[1] % 8) // 2
            y_offset = (像素.shape[2] % 8) // 2
            像素 = 像素[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
            蒙版 = 蒙版[:,:,x_offset:x + x_offset, y_offset:y + y_offset]

        #grow mask by a few pixels to keep things seamless in latent space
        if 扩大蒙版 == 0:
            mask_erosion = 蒙版
        else:
            kernel_tensor = torch.ones((1, 1, 扩大蒙版, 扩大蒙版))
            padding = math.ceil((扩大蒙版 - 1) / 2)

            mask_erosion = torch.clamp(torch.nn.functional.conv2d(蒙版.round(), kernel_tensor, padding=padding), 0, 1)

        m = (1.0 - 蒙版.round()).squeeze(1)
        for i in range(3):
            像素[:,:,:,i] -= 0.5
            像素[:,:,:,i] *= m
            像素[:,:,:,i] += 0.5
        t = VAE模型.encode(像素)

        return ({"samples":t, "noise_mask": (mask_erosion[:,:,:x,:y].round())}, )

class 批次选择_Zho:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"潜空间图像": ("LATENT",),
                             "批次编号": ("INT", {"default": 0, "min": 0, "max": 63}),
                             "长度": ("INT", {"default": 1, "min": 1, "max": 64}),
                             }}
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("潜空间",)
    FUNCTION = "frombatch"

    CATEGORY = "Zho汉化模块组/潜空间"

    def frombatch(self, 潜空间图像, 批次编号, 长度):
        s = 潜空间图像.copy()
        s_in = 潜空间图像["samples"]
        批次编号 = min(s_in.shape[0] - 1, 批次编号)
        长度 = min(s_in.shape[0] - 批次编号, 长度)
        s["samples"] = s_in[批次编号:批次编号 + 长度].clone()
        if "noise_mask" in 潜空间图像:
            masks = 潜空间图像["noise_mask"]
            if masks.shape[0] == 1:
                s["noise_mask"] = masks.clone()
            else:
                if masks.shape[0] < s_in.shape[0]:
                    masks = masks.repeat(math.ceil(s_in.shape[0] / masks.shape[0]), 1, 1, 1)[:s_in.shape[0]]
                s["noise_mask"] = masks[批次编号:批次编号 + 长度].clone()
        if "batch_index" not in s:
            s["batch_index"] = [x for x in range(批次编号, 批次编号 + 长度)]
        else:
            s["batch_index"] = 潜空间图像["batch_index"][批次编号:批次编号 + 长度]
        return (s,)

class 批次复制_Zho:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"潜空间图像": ("LATENT",),
                             "数量": ("INT", {"default": 1, "min": 1, "max": 64}),
                             }}
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("潜空间",)
    FUNCTION = "repeat"

    CATEGORY = "Zho汉化模块组/潜空间"

    def repeat(self, 潜空间图像, 数量):
        s = 潜空间图像.copy()
        s_in = 潜空间图像["samples"]

        s["samples"] = s_in.repeat((数量, 1, 1, 1))
        if "noise_mask" in 潜空间图像 and 潜空间图像["noise_mask"].shape[0] > 1:
            masks = 潜空间图像["noise_mask"]
            if masks.shape[0] < s_in.shape[0]:
                masks = masks.repeat(math.ceil(s_in.shape[0] / masks.shape[0]), 1, 1, 1)[:s_in.shape[0]]
            s["noise_mask"] = 潜空间图像["noise_mask"].repeat((数量, 1, 1, 1))
        if "batch_index" in s:
            offset = max(s["batch_index"]) - min(s["batch_index"]) + 1
            s["batch_index"] = s["batch_index"] + [x + (i * offset) for i in range(1, 数量) for x in s["batch_index"]]
        return (s,)

class 潜空间放大_Zho:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",), "放大方法": (s.upscale_methods,),
                              "宽度": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                              "高度": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                              "剪裁": (s.crop_methods,)}}
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("潜空间",)
    FUNCTION = "upscale"

    CATEGORY = "Zho汉化模块组/潜空间"

    def upscale(self, samples, 放大方法, 宽度, 高度, 剪裁):
        s = samples.copy()
        s["samples"] = comfy.utils.common_upscale(samples["samples"], 宽度 // 8, 高度 // 8, 放大方法, 剪裁)
        return (s,)

#latent放大模型
class 潜空间放大_比例_Zho:
    放大方法 = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples": ("LATENT",),
                             "放大方法": (s.放大方法,),
                             "比例": ("FLOAT", {"default": 1.5, "min": 0.01, "max": 8.0, "step": 0.01}),}}
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("潜空间",)
    FUNCTION = "upscale"

    CATEGORY = "Zho汉化模块组/潜空间"

    def upscale(self, samples, 放大方法, 比例):
        s = samples.copy()
        宽度 = round(samples["samples"].shape[3] * 比例)
        高度 = round(samples["samples"].shape[2] * 比例)
        s["samples"] = comfy.utils.common_upscale(samples["samples"], 宽度, 高度, 放大方法, "disabled")
        return (s,)

#---------------------------------
#图像
class 图像保存_Zho:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.disable_metadata = False  # 添加一个属性来控制是否禁用元数据

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"图像": ("IMAGE", ),
                     "文件名前缀": ("STRING", {"default": "ComfyUI"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "Zho汉化模块组/图像"

    def save_images(self, 图像, 文件名前缀="ComfyUI", prompt=None, extra_pnginfo=None):
        文件名前缀 += self.prefix_append
        full_output_folder, filename, counter, subfolder, 文件名前缀 = folder_paths.get_save_image_path(文件名前缀, self.output_dir, 图像[0].shape[1], 图像[0].shape[0])
        results = list()
        for image in 图像:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not self.disable_metadata:  # 使用 self.disable_metadata 判断是否禁用元数据
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return {"ui": {"images": results}}

class 图像预览_Zho(图像保存_Zho):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.disable_metadata = False  # 添加一个属性来控制是否禁用元数据

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"图像": ("IMAGE", ), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }


#加载图像
class 图像加载_Zho:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), )},
                }

    CATEGORY = "Zho汉化模块组/图像"

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("图像", "蒙版")
    FUNCTION = "load_image"
    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, mask)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True


#放大图像
class 图像放大_Zho:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"图像": ("IMAGE",), "放大方法": (s.upscale_methods,),
                             "宽度": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                             "高度": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                             "剪裁": (s.crop_methods,)}}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像", )
    FUNCTION = "upscale"

    CATEGORY = "Zho汉化模块组/图像"

    def upscale(self, 图像, 放大方法, 宽度, 高度, 剪裁):
        samples = 图像.movedim(-1,1)
        s = comfy.utils.common_upscale(samples, 宽度, 高度, 放大方法, 剪裁)
        s = s.movedim(1,-1)
        return (s,)

#放大图像-模型
class 图像放大_比例_Zho:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"图像": ("IMAGE",), "放大方法": (s.upscale_methods,),
                             "比例": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),}}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像", )
    FUNCTION = "upscale"

    CATEGORY = "Zho汉化模块组/图像"

    def upscale(self, 图像, 放大方法, 比例):
        samples = 图像.movedim(-1,1)
        宽度 = round(samples.shape[3] * 比例)
        高度 = round(samples.shape[2] * 比例)
        s = comfy.utils.common_upscale(samples, 宽度, 高度, 放大方法, "disabled")
        s = s.movedim(1,-1)
        return (s,)

#反转图像
class 图像反转_Zho:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"图像": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像", )
    FUNCTION = "invert"

    CATEGORY = "Zho汉化模块组/图像"

    def invert(self, 图像):
        s = 1.0 - 图像
        return (s,)

#分类
NODE_CLASS_MAPPINGS = {
    "采样器_Zho": 采样器_Zho,
    "高级采样器_Zho": 高级采样器_Zho,
    "主模型加载器_Zho": 主模型加载器_Zho,
    "VAE加载器_Zho": VAE加载器_Zho,
    "Lora加载器_Zho": Lora加载器_Zho,
    "ControlNet加载器_Zho": ControlNet加载器_Zho,
    "GLIGEN加载器_Zho": GLIGEN加载器_Zho,
    "提示词_Zho": 提示词_Zho,
    "CLIP跳过层_Zho": CLIP跳过层_Zho,
    "GLIGEN区域设定_Zho": GLIGEN区域设定_Zho,
    "ControlNet_Zho": ControlNet_Zho,
    "初始潜空间_Zho": 初始潜空间_Zho,
    "VAE解码器_Zho": VAE解码器_Zho,
    "VAE编码器_Zho": VAE编码器_Zho,
    "VAE编码器_重绘_Zho": VAE编码器_重绘_Zho,
    "批次选择_Zho": 批次选择_Zho,
    "批次复制_Zho": 批次复制_Zho,
    "潜空间放大_Zho": 潜空间放大_Zho,
    "潜空间放大_比例_Zho": 潜空间放大_比例_Zho,
    "图像保存_Zho": 图像保存_Zho,
    "图像预览_Zho": 图像预览_Zho,
    "图像加载_Zho": 图像加载_Zho,
    "图像放大_Zho": 图像放大_Zho,
    "图像放大_比例_Zho": 图像放大_比例_Zho,
    "图像反转_Zho": 图像反转_Zho,
}
