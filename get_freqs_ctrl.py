# from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from diffusers.utils import make_image_grid, load_image
from daam import set_seed
from transformers import pipeline
import numpy as np

from tqdm import tqdm

from autoencoder import *
from training_pipeline import AnalysisPipeline

assert not cfg['training'], "Disable training in config"

ae_version = 15
sub_folder = r'ctrl_freqs/'
cycles_per_image = 2000
images_to_test = [r'../imgs/table_with_edges.png', r'../imgs/table_with_cup_edge.png', r'../imgs/table_with_hat_edges.png']
freqs_output_postfix = [r'_empty', r'_cup', r'_hat']
prompt = 'A table in the kitchen'

ae_folders = os.listdir(cfg['save_dir'])
dims = wrapper.get_sizes()
autoencoders = []
all_freqs = []
totals = []

for i in range(0, len(ae_folders)) :
    ae = AutoEncoder(cfg, dims[i], i)
    ae = ae.load(ae_version)
    autoencoders.append(ae)
    all_freqs.append(np.zeros(ae.d_hidden))
    totals.append(0)


for img_path, img_save_postfix in zip(images_to_test, freqs_output_postfix):
    wrapper.set_image(img_path)
    buffers_handler = BuffersHandler(cfg)
    for _ in tqdm(range(0, cycles_per_image)) :
        all_acts = buffers_handler.next()
        for i in range (0, len(autoencoders)) :
            hidden = autoencoders[i](all_acts[i])[2]
            hidden_cpu = hidden.detach().cpu().numpy()

            all_freqs[i] += (hidden_cpu > 0).sum(0)
            totals[i] += hidden_cpu.shape[0]
    
    for i in range (0, len(all_freqs)) :
        all_freqs[i] /= totals[i]
        os.makedirs(os.path.join(cfg['save_dir'], f'ae_{i}', sub_folder), exist_ok=True)
        np.save(os.path.join(cfg['save_dir'], f'ae_{i}', sub_folder, 'freqs' + img_save_postfix + '.npy'), all_freqs[i])
        all_freqs[i] = np.zeros_like(all_freqs[i])
        totals[i] = 0

        


            