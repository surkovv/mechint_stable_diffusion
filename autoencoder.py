
import json
import pprint
import torch
import numpy as np
import random
from pathlib import Path
from torch import nn
from training_pipeline import Wrapper
import os
import torch.nn.functional as F


cfg = {
    "vector_len" : 320,
    "batch_vector_num" : 500,
    "batch_num" : 8,

    "d_hidden_mul" : 256,
    "l1_coeff" : 3e-4,
    "save_dir" : "./checkpoints/",

    "device" : "cuda:0",
    "seed" : 49
}
cfg["batch_size"] = cfg["vector_len"] * cfg["batch_vector_num"]
cfg["buffer_size"] = cfg["batch_size"] * cfg["batch_num"]

cfg["d_hidden"] = cfg["vector_len"] * cfg["d_hidden_mul"]

def get_acts(): # get output from mlp layer
    return torch.zeros((cfg["batch_vector_num"], cfg["vector_len"]))

wrapper = Wrapper()

class Buffer():
    def __init__(self, cfg):
        self.buffer = None
        self.cfg = cfg
        self.token_pointer = 0
        self.refresh()
    
    @torch.no_grad()
    def refresh(self):
        self.pointer = 0
        acts = wrapper.get()
        if acts == None:
            self.pointer = -1
            return
        acts = acts.flatten().to(cfg['device'])
        self.buffer = acts

        self.pointer = 0

    @torch.no_grad()
    def next(self):
        out = self.buffer[self.pointer:self.pointer+self.cfg["batch_size"]]
        out = torch.reshape(out, (cfg["batch_vector_num"], cfg["vector_len"]))
        self.pointer += self.cfg["batch_size"]
        if self.pointer >= self.buffer.shape[0]:
            self.refresh()
            if self.pointer == -1:
                return None
        return out


class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        torch.manual_seed(cfg["seed"])
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg["vector_len"], cfg["d_hidden"], dtype=torch.float32)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg["d_hidden"], cfg["vector_len"], dtype=torch.float32)))
        self.b_enc = nn.Parameter(torch.zeros(cfg["d_hidden"], dtype=torch.float32))
        self.b_dec = nn.Parameter(torch.zeros(cfg["vector_len"], dtype=torch.float32))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.to(cfg["device"])
    
    def forward(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = cfg["l1_coeff"] * (acts.float().abs().sum())
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss
    
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        # Bugfix(?) for ensuring W_dec retains unit norm, this was not there when I trained my original autoencoders.
        self.W_dec.data = W_dec_normed
    
    def get_version(self):
        version_list = [int(file.split(".")[0]) for file in list(os.listdir(cfg["save_dir"])) if "pt" in str(file)]
        if len(version_list):
            return 1+max(version_list)
        else:
            return 0

    def save(self):
        version = self.get_version()
        torch.save(self.state_dict(), os.path.join(cfg["save_dir"], str(version)+".pt"))
        with open(os.path.join(cfg["save_dir"], str(version)+"_cfg.json"), "w") as f:
            json.dump(cfg, f)
        print("Saved as version", version)
    
    @classmethod
    def load(cls, version):
        cfg = (json.load(open(os.path.join(cfg["save_dir"], str(version)+"_cfg.json"), "r")))
        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(torch.load(os.path.join(cfg["save_dir"], str(version)+".pt")))
        return self

