import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter

class ImageToContrastMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "low_threshold": ("INT", {
                    "default": 1, 
                    "min": 1,
                    "max": 255,
                    "step": 1
                }),
                "high_threshold": ("INT", {
                    "default": 255, 
                    "min": 1,
                    "max": 255,
                    "step": 1
                }),
                "blur_radius": ("INT", {
                    "default": 1, 
                    "min": 1,
                    "max": 32768,
                    "step": 1
                })
            },
        }

    RETURN_TYPES = ("IMAGE","MASK",)
    FUNCTION = "image_to_contrast_mask"

    CATEGORY = "XSS"

    def image_to_contrast_mask(self, image, low_threshold, high_threshold, blur_radius):
        image = 255. * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        image = ImageOps.grayscale(image)

        if blur_radius > 1:
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        #high_filter = lambda x: 255 if x > high_threshold else x
        #image = image.convert("L").point(high_filter, mode="L")

        #low_filter = lambda x: 0 if x < low_threshold else x
        #image = image.convert("L").point(high_filter, mode="L")

        filter = lambda x: 255 if x > high_threshold else 0 if x < low_threshold else x
        image = image.convert("L").point(filter, mode="L")

        image = np.array(image).astype(np.float32) / 255.0
        mask = 1. - torch.from_numpy(image)
        image = torch.from_numpy(image)[None,]
        
        return (image, mask,)

NODE_CLASS_MAPPINGS = {
    "ImageToContrastMask": ImageToContrastMask
}