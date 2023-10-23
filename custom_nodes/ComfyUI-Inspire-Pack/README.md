# ComfyUI-Inspire-Pack
This repository offers various extension nodes for ComfyUI. Nodes here have different characteristics compared to those in the ComfyUI Impact Pack. The Impact Pack has become too large now...

## Notice:
* V0.13.2 isn't compatible with old ControlNet Auxiliary Preprocessor. If you will use `MediaPipeFaceMeshDetectorProvider` update to latest version(Sep. 17th).
* WARN: If you use version **0.12 to 0.12.2** without a GlobalSeed node, your workflow's seed may have been erased. Please update immediately.

## Nodes
* Lora Block Weight - This is a node that provides functionality related to Lora block weight.
    * This provides similar functionality to [sd-webui-lora-block-weight](https://github.com/hako-mikan/sd-webui-lora-block-weight)
    * `Lora Loader (Block Weight)`: When loading Lora, the block weight vector is applied.
        * In the block vector, you can use numbers, R, A, a, B, and b.
        * R is determined sequentially based on a random seed, while A and B represent the values of the A and B parameters, respectively. a and b are half of the values of A and B, respectively.
    * `XY Input: Lora Block Weight`: This is a node in the [Efficiency Nodes](https://github.com/LucianoCirino/efficiency-nodes-comfyui)' XY Plot that allows you to use Lora block weight.
        * You must ensure that X and Y connections are made, and dependencies should be connected to the XY Plot.
        * Note: To use this feature, update `Efficient Nodes` to a version released after September 3rd.

* SEGS Supports nodes - This is a node that supports ApplyControlNet (SEGS) from the Impact Pack.
    * `OpenPose Preprocessor Provider (SEGS)`: OpenPose preprocessor is applied for the purpose of using OpenPose ControlNet in SEGS.
        * You need to install [ControlNet Auxiliary Preprocessors](https://github.com/Fannovel16/comfyui_controlnet_aux) to use this.
    * `Canny Preprocessor Provider (SEGS)`: Canny preprocessor is applied for the purpose of using Canny ControlNet in SEGS.
    * `DW Preprocessor Provider (SEGS)`, `MiDaS Depth Map Preprocessor Provider (SEGS)`, `LeReS Depth Map Preprocessor Provider (SEGS)`, 
      `MediaPipe FaceMesh Preprocessor Provider (SEGS)`, `HED Preprocessor Provider (SEGS)`, `Fake Scribble Preprocessor (SEGS)`, 
      `AnimeLineArt Preprocessor Provider (SEGS)`, `Manga2Anime LineArt Preprocessor Provider (SEGS)`, `LineArt Preprocessor Provider (SEGS)`,
      `Color Preprocessor Provider (SEGS)`, `Inpaint Preprocessor Provider (SEGS)`, 
    * `MediaPipeFaceMeshDetectorProvider`: This node provides `BBOX_DETECTOR` and `SEGM_DETECTOR` that can be used in Impact Pack's Detector using the `MediaPipe-FaceMesh Preprocessor` of ControlNet Auxiliary Preprocessors.

* A1111 Compatibility support - These nodes assists in replicating the creation of A1111 in ComfyUI exactly.
    * `KSampler (Inspire)`: ComfyUI uses the CPU for generating random noise, while A1111 uses the GPU. One of the three factors that significantly impact reproducing A1111's results in ComfyUI can be addressed using `KSampler (Inspire)`.
        * Other point #1 : Please make sure you haven't forgotten to include 'embedding:' in the embedding used in the prompt, like 'embedding:easynegative.'
        * Other point #2 : ComfyUI and A1111 have different interpretations of weighting. To align them, you need to use [BlenderNeko/Advanced CLIP Text Encode](https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb).
    * `KSamplerAdvanced (Inspire)`: Inspire Pack version of `KSampler (Advanced)`.

* Prompt Support - These are nodes for supporting prompt processing.
  * `Load Prompts From Dir (Inspire)`: It sequentially reads prompts files from the specified directory. The output it returns is ZIPPED_PROMPT.
    * Specify the directories located under `ComfyUI-Inspire-Pack/prompts/`
    * One prompts file can have multiple prompts separated by `---`. 
    * e.g. `prompts/example`
  * `Load Prompts From File (Inspire)`: It sequentially reads prompts from the specified file. The output it returns is ZIPPED_PROMPT.
    * Specify the file located under `ComfyUI-Inspire-Pack/prompts/`
    * e.g. `prompts/example/prompt2.txt` 
  * `Unzip Prompt (Inspire)`: Separate ZIPPED_PROMPT into `positive`, `negative`, and name components. 
    * `positive` and `negative` represent text prompts, while `name` represents the name of the prompt. When loaded from a file using `Load Prompts From File (Inspire)`, the name corresponds to the file name.
  * `Zip Prompt (Inspire)`: Create ZIPPED_PROMPT from positive, negative, and name_opt.
    * If name_opt is omitted, it will be considered as an empty name.
  * `Prompt Extractor (Inspire)`: This node reads prompt information from the image's metadata. Since it retrieves all the text, you need to directly specify the prompts to be used for `positive` and `negative` as indicated in the info.
  * `Global Seed (Inspire)`: This is a node that controls the global seed without a separate connection line. It only controls when the widget's name is 'seed' or 'noise_seed'. Additionally, if 'control_before_generate' is checked, it controls the seed before executing the prompt.
    * Seeds that have been converted into inputs are excluded from the target. If you want to control the seed separately, convert it into an input and control it separately.
  * `Bind [ImageList, PromptList] (Inspire)`: Bind Image list and zipped prompt list to export `image`, `positive`, `negative`, and `prompt_label` in a list format. If there are more prompts than images, the excess prompts are ignored, and if there are not enough, the remainder is filled with default input based on the images.
  * `Wildcard Encode (Inspire)`: The combination node of [ImpactWildcardEncode](https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/ImpactWildcard.md) and BlenderNeko's [CLIP Text Encode (Advanced)](https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb).
    * To use this node, you need both the [Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) and the [Advanced CLIP Text Encode]((https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb)) extensions.
    * This node is identical to `ImpactWildcardEncode`, but it encodes using `CLIP Text Encode (Advanced)` instead of the default CLIP Text Encode from ComfyUI for CLIP Text Encode.
    * Requirement: Impact Pack V4.18.6 or above

* Regional Sampler - These nodes assists in the easy utilization of the regional sampler in the `Impact Pack`.
  * `Regional Prompt Simple (Inspire)`: This node takes `mask` and `basic_pipe` as inputs and simplifies the creation of `REGIONAL_PROMPTS`.
  * `Regional Prompt By Color Mask (Inspire)`: Similar to `Regional Prompt Simple (Inspire)`, this function accepts a color mask image as input and defines the region using the color value that will be used as the mask, instead of directly receiving the mask.
    * The color value can only be in the form of a hex code like #FFFF00 or a decimal number. 

* Image Util
  * `Load Images From Dir (Inspire)`: This is almost same as `LoadImagesFromDirectory` of [ComfyUI-Advanced-Controlnet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet). This is just a modified version. Just note that this node forcibly normalizes the size of the loaded image to match the size of the first image, even if they are not the same size, to create a batch image.  
  * `Load Image (Inspire)`: This node is similar to LoadImage, but the loaded image information is stored in the workflow. The image itself is stored in the workflow, making it easier to reproduce image generation on other computers.
  
* KSampler Progress - In KSampler, the sampling process generates latent batches. By using `Video Combine` node from [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite), you can create a video from the progress.

* Backend Cache - Nodes for storing arbitrary data from the backend in a cache and sharing it across multiple workflows.
  * `Cache Backend Data (Inspire)`: Stores any backend data in the cache using a string key. Tags are for quick reference.
  * `Retrieve Backend Data (Inspire)`: Retrieves cached backend data using a string key.
  * `Remove Backend Data (Inspire)`: Removes cached backend data. 
    * Deletion in this node only removes it from the cache managed by Inspire, and if it's still in use elsewhere, it won't be completely removed from memory.
    * `signal_opt` is used to control the order of execution for this node; it will still run without a `signal_opt` input.
    * When using '*' as the key, it clears all data.
  * `Show Cached Info (Inspire)`: Displays information about cached data.
  * `Cache Backend Data [NumberKey] (Inspire)`, `Retrieve Backend Data [NumberKey] (Inspire)`, `Remove Backend Data [NumberKey] (Inspire)`: These nodes are provided for convenience in the automation process, allowing the use of numbers as keys.
   
## Credits

ComfyUI/[ComfyUI](https://github.com/comfyanonymous/ComfyUI) - A powerful and modular stable diffusion GUI.

ComfyUI/[sd-webui-lora-block-weight](https://github.com/hako-mikan/sd-webui-lora-block-weight) - The original idea for LoraBlockWeight came from here, and it is based on the syntax of this extension.

LucianoCirino[efficiency-nodes-comfyui](https://github.com/LucianoCirino/efficiency-nodes-comfyui) - The `XY Input` provided by the Inspire Pack supports the `XY Plot` of this node.

Fannovel16/[comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) - The wrapper for the controlnet preprocessor in the Inspire Pack depends on these nodes.

Kosinkadink/[ComfyUI-Advanced-Controlnet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet) - `Load Images From Dir (Inspire)` code is came from here. 