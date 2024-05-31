import numpy as np
from tqdm import tqdm
import argparse

from autoencoder import *
from training_pipeline import AnalysisPipeline
from cfg import cfg
from prompts import prompts

parser=argparse.ArgumentParser()
parser.add_argument("img_path", type=str)
parser.add_argument("save_subfolder", type=str)
parser.add_argument("--ae_version", default=15, type=int)
parser.add_argument("--cycles", default=300, type=int)
args=parser.parse_args()

wrapper = AnalysisPipeline(restrict=list(range(4, 13)))

ae_folders = os.listdir(cfg['save_dir'])
dims = wrapper.get_sizes()
autoencoders = []
all_freqs = []
totals = []

for i in range(0, len(ae_folders)) :
    ae = AutoEncoder(cfg, dims[i], i)
    ae = ae.load(cfg, args.ae_version)
    autoencoders.append(ae)
    all_freqs.append(np.zeros(ae.d_hidden))
    totals.append(0)


for prompt, name in prompts:
    wrapper.set_image(args.img_path)
    wrapper.set_prompt(prompt)
    buffers_handler = BuffersHandler(cfg, wrapper)
    for _ in tqdm(range(0, args.cycles)) :
        all_acts = buffers_handler.next()
        for i in range (0, len(autoencoders)) :
            hidden = autoencoders[i](all_acts[i])[2]
            hidden_cpu = hidden.detach().cpu().numpy()

            all_freqs[i] += (hidden_cpu > 0).sum(0)
            totals[i] += hidden_cpu.shape[0]
    
    for i in range (0, len(all_freqs)) :
        all_freqs[i] /= totals[i]
        np.save(os.path.join(cfg['save_dir'], f'ae_{i}', args.save_subfolder, 'freqs_' + name + '.npy'), all_freqs[i])
        os.makedirs(os.path.join(cfg['save_dir'], f'ae_{i}', args.save_subfolder), exist_ok=True)
        all_freqs[i] = np.zeros_like(all_freqs[i])
        totals[i] = 0

        


            