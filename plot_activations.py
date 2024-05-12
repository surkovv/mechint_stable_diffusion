import torch
import matplotlib.pyplot as plt
import numpy as np
import json

prompts = json.load(open('../prompts.json', 'r'))
activations = np.load('../activations.npy')
freqs = np.load('../freqs.npy')

is_norm = freqs > -2.5
for i in range(100):
    if is_norm[i]:
        print(i)

norm_activations = activations[:, is_norm]

def plot_feature(feature_num):
    acts = norm_activations[:, feature_num].squeeze()
    pas = list(zip(prompts, acts))
    pas.sort(key=lambda x: x[1], reverse=True)
    for p, a in pas[:10]:
        print(p, a)

def get_best_features(word):
    def get_avg_place(act):
        places = np.argsort(act)[::-1]
        sm = 0
        tot = 0
        for i in range(len(places)):
            if word in prompts[places[i]].lower():
                sm += i
                tot += 1
        return sm / tot
    avg_places = np.array([
        get_avg_place(norm_activations[:, i]) for i in range(norm_activations.shape[1])
    ])
    print(avg_places.min())
    return avg_places.argmin()

f = get_best_features('kitchen')
print(f)
# print(norm_activations.max(axis=0).topk(10))
plot_feature(f)