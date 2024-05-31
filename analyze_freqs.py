import numpy as np
import matplotlib.pyplot as plt
import argparse

from cfg import cfg, thresholds

parser=argparse.ArgumentParser()
parser.add_argument("ae_number", type=int)
parser.add_argument("freqs_name", type=str)
parser.add_argument("freqs_for_comparing_name", type=str)
parser.add_argument("--mult_threshold", default=100, type=int)
parser.add_argument("--subfolder", default=r'ctrl_freqs/', type=str)
parser.add_argument("--plot_savedir", default=r'analyzes/freqs_hist.png', type=str)
parser.add_argument("--ae_version", default=15, type=int)
args=parser.parse_args()

freqs_path = cfg["save_dir"] + r'ae_' + str(args.ae_number) + r'/' + args.subfolder
freqs1 = np.load(freqs_path + args.freqs_name)
freqs2 = np.load(freqs_path + args.freqs_for_comparing_name)

all_neurons = np.arange(0, freqs1.shape[-1])
assert freqs1.shape[-1] == freqs2.shape[-1], "Sizes of freqs should be equal"

offset = 10 ** -6
active_neurons = []
for i in range(0, freqs1.shape[-1]) :
    if freqs1[i] > freqs2[i] * args.mult_threshold and np.log10(freqs1[i] + offset) > thresholds[args.ae_number] :
        active_neurons.append(i)

freqs_original = np.load(cfg['save_dir'] + r'ae_' + str(args.ae_number) + r'/freqs_' + str(args.ae_version) + '.npy')
freqs_original_log = np.log10(freqs_original + offset)
plt.hist(freqs_original_log, bins=30)
active_freqs_log = np.log10(freqs1[active_neurons] + offset)
print("Active neurons: ", active_neurons)
for i, freq in enumerate(active_freqs_log) :
    plt.vlines(freq, ymin=0, ymax=3000, color='k')
    plt.text(freq, 3000, f'{active_neurons[i]}', rotation=90, verticalalignment='bottom', horizontalalignment='right', fontsize=8)

plt.savefig(args.plot_savedir)
