# from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from diffusers.utils import make_image_grid, load_image
from daam import set_seed
from transformers import pipeline
import numpy as np

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

path = '../imgs/table_with_edges.png'
prompts = [('a table in a kitchen', 'table'),
            ('a glass table in a kitchen', 'glass'),
            ('a iron table in a kitchen', 'iron'),
            ('a table in a kitchen with tile floor', 'tile'),
            ('a table in a kitchen with laminate floor', 'laminate'),
]

for prompt, name in prompts:
    loaded_img = load_image(path)
    out = pipe(prompt, loaded_img, num_inference_steps=15,
                num_images_per_prompt=3, gen = set_seed(0))
    

    results = out.images

    for i in range(0, len(results)) :
        results[i].save(f'out_imgs/{i}_{name}.png')