from datasets import load_dataset
from daam import trace, set_seed, UNetFFLocator, trace_intervention
from diffusers import DiffusionPipeline
from matplotlib import pyplot as plt
import torch
from random import shuffle
from diffusers import StableDiffusionImg2ImgPipeline, ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from diffusers.utils import make_image_grid, load_image
import numpy as np
from tqdm import tqdm
from autoencoder import *

ae_version = 15
ae_folders = os.listdir(cfg['save_dir'])
dims = wrapper.get_sizes()
autoencoders = []

for i in range(0, len(ae_folders)) :
    ae = AutoEncoder(cfg, dims[i], i)
    ae = ae.load(ae_version)
    autoencoders.append(ae)

pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, safety_checker=None).to('cuda')
prompt = "A green table in the kitchen"

def generate_image(pipe, out_name):
    out = pipe(prompt, num_inference_steps=15, num_images_per_prompt=10, generator=set_seed(0))
    for i, img in enumerate(out.images):
        img.save('gen_imgs/' + out_name + f'_{i}.png')
    
generate_image(pipe, 'no_intervention')

my_neurons = {
    1: [39288],
    3: [14093],
    5: [14598]
}

reset_masks = []
for i in range(len(autoencoders)):
    if i not in my_neurons:
        reset_masks.append(None)
    else:
        mask = torch.zeros(autoencoders[i].d_hidden, dtype=torch.bool).to('cuda')
        for neuron in my_neurons[i]:
            mask[neuron] = 1
        reset_masks.append(mask)

with trace_intervention(pipeline=pipe, autoencoders=autoencoders, reset_masks=reset_masks) as trace:
    generate_image(pipe, 'intervention')
    

