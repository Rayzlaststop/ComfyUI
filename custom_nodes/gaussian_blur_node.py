import torch
import numpy as np
from PIL import Image, ImageFilter 

class GaussianBlur:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "radius": ("INT", {
                    "default": 4, 
                    "min": 1,
                    "max": 1024,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "gaussian_blur"

    CATEGORY = "XSS"

    def gaussian_blur(self, image, radius):
        image = 255. * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        filtered_image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        image = filtered_image.convert("RGB")
        filtered_image = np.array(filtered_image).astype(np.float32) / 255.0
        filtered_image = torch.from_numpy(filtered_image)[None,]
        
        return (filtered_image,)

NODE_CLASS_MAPPINGS = {
    "GaussianBlur": GaussianBlur
}