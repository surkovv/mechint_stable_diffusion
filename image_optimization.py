from datasets import load_dataset
from daam import trace, set_seed, UNetFFLocator
from diffusers import DiffusionPipeline
from matplotlib import pyplot as plt
import torch
from random import shuffle
from diffusers import StableDiffusionImg2ImgPipeline, ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from diffusers.utils import make_image_grid, load_image
import numpy as np
from tqdm import tqdm

pipe = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, safety_checker=None).to('cuda')

prompt = "a table in a kitchen"
gen = set_seed(0) 

image_opt = torch.rand(3, 256, 256)
image_opt.requires_grad = True
optimizer = torch.optim.Adam([image_opt], lr=0.1)

ae_num = 4
act_num = 100

for i in tqdm(range(100)):
    optimizer.zero_grad()
    # image_opt = image_opt.clamp(0, 1)
    
    result = {}
    with trace(pipe, result, do_rand=True) as tc:
        out = pipe(prompt, image_opt, num_inference_steps=15, num_images_per_prompt=3, generator=gen)

    if i % 10 == 0:
        for j, img in enumerate(out.images):
            img.save(f'gen_imgs/iter_{i}_n_{j}.png')

    acts = result[ae_num]

    acts = torch.cat(acts).flatten(0, 1)

    loss = -acts[:, act_num].mean()
    
    loss.backward()
    optimizer.step()