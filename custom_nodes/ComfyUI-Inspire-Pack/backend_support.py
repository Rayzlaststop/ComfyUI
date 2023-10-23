from .libs.utils import any_typ
from server import PromptServer

cache = {}


class CacheBackendData:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("STRING", {"multiline": False, "placeholder": "Input data key (e.g. 'model a', 'chunli lora', 'girl latent 3', ...)"}),
                "tag": ("STRING", {"multiline": False, "placeholder": "Tag: short description"}),
                "data": (any_typ,),
            }
        }

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("data opt",)

    FUNCTION = "doit"

    CATEGORY = "InspirePack/Backend"

    OUTPUT_NODE = True

    def doit(self, key, tag, data):
        global cache

        if key == '*':
            print(f"[Inspire Pack] CacheBackendData: '*' is reserved key. Cannot use that key")

        cache[key] = (tag, data)
        return (data,)


class CacheBackendDataNumberKey:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "tag": ("STRING", {"multiline": False, "placeholder": "Tag: short description"}),
                "data": (any_typ,),
            }
        }

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("data opt",)

    FUNCTION = "doit"

    CATEGORY = "InspirePack/Backend"

    OUTPUT_NODE = True

    def doit(self, key, tag, data):
        global cache
        cache[key] = (tag, data)
        return (data,)



class RetrieveBackendData:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("STRING", {"multiline": False, "placeholder": "Input data key (e.g. 'model a', 'chunli lora', 'girl latent 3', ...)"}),
            }
        }

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("data",)

    FUNCTION = "doit"

    CATEGORY = "InspirePack/Backend"

    def doit(self, key):
        global cache

        return (cache[key][1],)


class RetrieveBackendDataNumberKey(RetrieveBackendData):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }


class RemoveBackendData:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("STRING", {"multiline": False, "placeholder": "Input data key ('*' = clear all)"}),
            },
            "optional": {
                "signal_opt": (any_typ,),
            }
        }

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("signal",)

    FUNCTION = "doit"

    CATEGORY = "InspirePack/Backend"

    OUTPUT_NODE = True

    def doit(self, key, signal_opt=None):
        global cache

        if key == '*':
            cache = {}
        elif key in cache:
            del cache[key]
        else:
            print(f"[Inspire Pack] RemoveBackendData: invalid data key {key}")

        return (signal_opt,)


class RemoveBackendDataNumberKey(RemoveBackendData):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "signal_opt": (any_typ,),
            }
        }

    def doit(self, key, signal_opt=None):
        global cache

        if key in cache:
            del cache[key]
        else:
            print(f"[Inspire Pack] RemoveBackendDataNumberKey: invalid data key {key}")

        return (signal_opt,)


class ShowCachedInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cache_info": ("STRING", {"multiline": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ()

    FUNCTION = "doit"

    CATEGORY = "InspirePack/Backend"

    OUTPUT_NODE = True

    def doit(self, cache_info, unique_id):
        global cache

        text1 = "---- [String Key Caches] ----\n"
        text2 = "---- [Number Key Caches] ----\n"
        for k, v in cache.items():
            if v[0] == '':
                tag = 'N/A(tag)'
            else:
                tag = v[0]

            if isinstance(k, str):
                text1 += f'{k}: {tag}\n'
            else:
                text2 += f'{k}: {tag}\n'

        text = text1 + "\n" + text2
        PromptServer.instance.send_sync("inspire-node-feedback", {"id": unique_id, "widget_name": "cache_info", "type": "text", "data": text})

        return ()

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


NODE_CLASS_MAPPINGS = {
    "CacheBackendData //Inspire": CacheBackendData,
    "CacheBackendDataNumberKey //Inspire": CacheBackendDataNumberKey,
    "RetrieveBackendData //Inspire": RetrieveBackendData,
    "RetrieveBackendDataNumberKey //Inspire": RetrieveBackendDataNumberKey,
    "RemoveBackendData //Inspire": RemoveBackendData,
    "RemoveBackendDataNumberKey //Inspire": RemoveBackendDataNumberKey,
    "ShowCachedInfo //Inspire": ShowCachedInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CacheBackendData //Inspire": "Cache Backend Data (Inspire)",
    "CacheBackendDataNumberKey //Inspire": "Cache Backend Data [NumberKey] (Inspire)",
    "RetrieveBackendData //Inspire": "Retrieve Backend Data (Inspire)",
    "RetrieveBackendDataNumberKey //Inspire": "Retrieve Backend Data [NumberKey] (Inspire)",
    "RemoveBackendData //Inspire": "Remove Backend Data (Inspire)",
    "RemoveBackendDataNumberKey //Inspire": "Remove Backend Data [NumberKey] (Inspire)",
    "ShowCachedInfo //Inspire": "Show Cached Info (Inspire)",
}
