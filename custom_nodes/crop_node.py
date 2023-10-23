import torch
import numpy as np
from PIL import Image, ImageOps

class Crop:
    channels = ["red", "green", "blue", "greyscale"]
    modes = ["binary", "inverse binary", "to zero", "inverse to zero", "truncate", "inverse truncate"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {
                    "default": 0, 
                    "min": 0,
                    "max": 32768,
                    "step": 1               
                }),
                "top": ("INT", {
                    "default": 0, 
                    "min": 0,
                    "max": 32768,
                    "step": 1                   
                }),
                "right": ("INT", {
                    "default": 1, 
                    "min": 1,
                    "max": 32768,
                    "step": 1                    
                }),
                "bottom": ("INT", {
                    "default": 1, 
                    "min": 1,
                    "max": 32768,
                    "step": 1                     
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop"

    CATEGORY = "XSS"

    def crop(self, image, left, right, top, bottom):
        image = 255. * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        image = image.crop((left, top, right, bottom))
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        
        return (image,)

NODE_CLASS_MAPPINGS = {
    "Crop": Crop
}