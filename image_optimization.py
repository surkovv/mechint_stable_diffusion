from datasets import load_dataset
from daam import trace, set_seed, UNetFFLocator
from diffusers import DiffusionPipeline
from matplotlib import pyplot as plt
import torch
from random import shuffle
from diffusers import AutoPipelineForImage2Image, ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from diffusers.utils import make_image_grid, load_image
import numpy as np
from tqdm import tqdm

pipe = AutoPipelineForImage2Image.from_pretrained("CompVis/stable-diffusion-v1-4")

prompt = "a table in a kitchen"

image_opt = torch.rand(3, 256, 256)
optimizer = torch.optim.Adam([image_opt], lr=0.01)
gen = set_seed(0) 

ae_num = 1
act_num = 1000

for i in tqdm(range(100)):
    optimizer.zero_grad()
    
    result = {}
    with trace(pipe, result, do_rand=True) as tc:
        out = pipe(prompt, image_opt, num_inference_steps=15, num_images_per_prompt=20, generator=gen)

    if i % 10 == 0:
        for j, img in enumerate(out.images):
            img.save(f'gen_imgs/iter_{i}_n_{j}.png')

    acts = result[ae_num]

    loss = -acts[:, act_num].mean()
    
    loss.backward()
    optimizer.step()