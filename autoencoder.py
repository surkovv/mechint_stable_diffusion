import json
import torch
from torch import nn
import os
import torch.nn.functional as F

class Buffer:
    def __init__(self, vector_len, cfg):
        self.buffer = None
        self.cfg = cfg
        self.pointer = 0
        self.vector_len = vector_len
        self.batch_size = self.vector_len * cfg["batch_vector_num"]
        self.finished = False
    
    @torch.no_grad()
    def refresh(self, acts):
        acts = acts.flatten().to(self.cfg['device'])
        self.buffer = acts
        self.pointer = 0
        self.finished = False

    @torch.no_grad()
    def next(self):
        out = self.buffer[self.pointer:self.pointer+self.batch_size]
        out = torch.reshape(out, (self.cfg["batch_vector_num"], self.vector_len))
        self.pointer += self.batch_size
        if self.pointer >= self.buffer.shape[0]:
            self.finished = True
        return out
    
class BuffersHandler:
    def __init__(self, cfg, wrapper):
        self.buffers = []
        self.vector_lens = wrapper.get_sizes()
        self.buffers_num = len(self.vector_lens)
        for vector_len in self.vector_lens :
            self.buffers.append(Buffer(vector_len, cfg))
        
        self.done = False
        self.wrapper = wrapper
        self.cfg = cfg
        self.refresh()
    
    @torch.no_grad()
    def refresh(self) :
        acts_list = self.wrapper.get()
        if acts_list == None:
            self.done = True
            return
        
        for i in range(0, len(acts_list)) :
            self.buffers[i].refresh(acts_list[i])

    @torch.no_grad()
    def next(self):
        out_list = []
        refresh = False
        for buffer in self.buffers :
            out = buffer.next()
            out = torch.reshape(out, (self.cfg["batch_vector_num"], buffer.vector_len))
            out_list.append(out)
            refresh = (refresh or buffer.finished)
        
        if refresh :
            self.refresh()
            if (self.done) :
                return None

        return out_list

class AutoEncoder(nn.Module):
    def __init__(self, cfg, dim, num):
        super().__init__()
        torch.manual_seed(cfg["seed"])
        self.d_hidden = cfg["d_hidden_mul"] * dim
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(dim, self.d_hidden, dtype=torch.float32)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.d_hidden, dim, dtype=torch.float32)))
        self.b_enc = nn.Parameter(torch.zeros(self.d_hidden, dtype=torch.float32))
        self.b_dec = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.num = num
        self.dim = dim
        self.save_dir = os.path.join(cfg['save_dir'], f'ae_{num}')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.to(cfg["device"])
        self.cfg = cfg
    
    def forward(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.cfg["l1_coeff"] * (acts.float().abs().sum())
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss
    
    @torch.no_grad()
    def encode(self, input) :
        
        return F.relu(input @ self.W_enc + self.b_enc)
    
    @torch.no_grad()
    def decode(self, hidden) :
        return hidden @ self.W_dec + self.b_dec

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed
    
    def get_version(self):
        version_list = [int(file.split(".")[0]) for file in list(os.listdir(self.save_dir)) if "pt" in str(file)]
        if len(version_list):
            return 1+max(version_list)
        else:
            return 0

    def save(self):
        version = self.get_version()
        torch.save(self.state_dict(), os.path.join(self.save_dir, str(version)+".pt"))
        with open(os.path.join(self.save_dir, str(version)+"_cfg.json"), "w") as f:
            json.dump(self.cfg, f)
        print("Saved as version", version)

    def load(self, cfg, version):
        self = AutoEncoder(cfg=cfg, dim=self.dim, num=self.num)
        self.load_state_dict(torch.load(os.path.join(self.save_dir, str(version)+".pt")))
        return self

