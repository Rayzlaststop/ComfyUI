from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import torch
import comfy.utils

MAX_RESOLUTION=8192

def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Zho排版模块
class Zho排版模块:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "底图": ("IMAGE",),
                "新图": ("IMAGE",),
                "新图尺寸调整": (["否", "自适应", "缩放", "自定义大小"],),
                "新图调整方式": (["nearest-exact", "bilinear", "area"],),
                "缩放": ("FLOAT", {"default": 1, "min": 0.01, "max": 16.0, "step": 0.1}),
                "自定义宽度": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 64}),
                "自定义高度": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 64}),
                "起点x坐标": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 8}),
                "起点y坐标": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 8}),
                "旋转": ("INT", {"default": 0, "min": -180, "max": 180, "step": 5}),
                "透明度": ("FLOAT", {"default": 0, "min": 0, "max": 100, "step": 5}),
            },
            "optional": {"蒙版": ("MASK",),}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)

    FUNCTION = "Zho_layout"
    CATEGORY = "Zho模块组"

    def Zho_layout(self, 底图, 新图, 新图尺寸调整, 新图调整方式, 缩放, 自定义宽度, 自定义高度, 起点x坐标, 起点y坐标, 旋转, 透明度, 蒙版=None):
        result = self.apply_overlay(tensor2pil(底图), 新图, 新图尺寸调整, 新图调整方式, 缩放, (int(自定义宽度), int(自定义高度)),
                                   (int(起点x坐标), int(起点y坐标)), int(旋转), 透明度, 蒙版)
        return (pil2tensor(result),)

    def apply_overlay(self, base, overlay, size_option, resize_method, rescale_factor, size, location, rotation, opacity, mask):

        # Check for different sizing options
        if size_option != "否":
            #Extract overlay size and store in Tuple "overlay_size" (WxH)
            overlay_size = overlay.size()
            overlay_size = (overlay_size[2], overlay_size[1])
            if size_option == "自适应":
                overlay_size = (base.size[0],base.size[1])
            elif size_option == "缩放":
                overlay_size = tuple(int(dimension * rescale_factor) for dimension in overlay_size)
            elif size_option == "自定义大小":
                overlay_size = (size[0], size[1])

            samples = overlay.movedim(-1, 1)
            overlay = comfy.utils.common_upscale(samples, overlay_size[0], overlay_size[1], resize_method, False)
            overlay = overlay.movedim(1, -1)
            
        overlay = tensor2pil(overlay)

         # Add Alpha channel to overlay
        overlay = overlay.convert('RGBA')
        overlay.putalpha(Image.new("L", overlay.size, 255))

        # If mask connected, check if the overlay image has an alpha channel
        if mask is not None:
            # Convert mask to pil and resize
            mask = tensor2pil(mask)
            mask = mask.resize(overlay.size)
            # Apply mask as overlay's alpha
            overlay.putalpha(ImageOps.invert(mask))

        # Rotate the overlay image
        overlay = overlay.rotate(rotation, expand=True)

        # Apply opacity on overlay image
        r, g, b, a = overlay.split()
        a = a.point(lambda x: max(0, int(x * (1 - opacity / 100))))
        overlay.putalpha(a)

        # Paste the overlay image onto the base image
        if mask is None:
            base.paste(overlay, location)
        else:
            base.paste(overlay, location, overlay)

        # Return the edited base image
        return base

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "Zho排版模块": Zho排版模块,
}