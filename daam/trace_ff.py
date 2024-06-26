from pathlib import Path
from typing import List, Type, Any, Dict, Tuple, Union
import math
import os
import pickle
import itertools

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.attention_processor import Attention
import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
import random

from .utils import cache_dir, auto_autocast
from .hook import ObjectHooker, AggregateHooker, UNetCrossAttentionLocator, ModuleLocator


class DiffusionFFHooker(AggregateHooker):
    def __init__(
            self,
            pipeline: Union[StableDiffusionPipeline, StableDiffusionXLPipeline],
            result: dict,
            do_rand: bool,
            restrict: list = list(range(4, 13))
    ):
        self.locator = UNetFFLocator()

        if restrict is None:
            restrict = list(range(100))

        modules = [
            UNetFFHooker(
                x,
                layer_idx=idx,
                result=result,
                do_rand=do_rand
            ) for idx, x in enumerate(self.locator.locate(pipeline.unet)) if idx in restrict
        ]

        self.pipe = pipeline
        self.restrict = restrict
        self.result = result
        super().__init__(modules)

    @property
    def layer_names(self):
        return self.locator.layer_names


class DiffusionFFHookerIntervent(AggregateHooker):
    def __init__(
        self,
        pipeline: Union[StableDiffusionPipeline, StableDiffusionXLPipeline],
        restrict: list = list(range(4, 13)),
        autoencoders = None,
        reset_masks: List[torch.Tensor] = None
    ):
        self.locator = UNetFFLocator()

        if restrict is None:
            restrict = list(range(100))

        xs = [x for idx, x in enumerate(self.locator.locate(pipeline.unet)) if idx in restrict]

        modules = [
            UNetFFHookerIntervent(
                x,
                layer_idx=idx,
                autoencoder=autoencoder,
                neurons_to_reset=reset_mask
            ) for idx, (x, autoencoder, reset_mask) in enumerate(
                zip(xs, autoencoders, reset_masks))
        ]

        self.pipe = pipeline
        super().__init__(modules)

    @property
    def layer_names(self):
        return self.locator.layer_names


class UNetFFHooker(ObjectHooker[Attention]):
    def __init__(
        self,
        module,
        layer_idx,
        result,
        do_rand
    ):
        super().__init__(module)
        self.layer_idx = layer_idx
        self.data = []
        self.result = result
        self.do_rand = do_rand

    def forw(self, module, inp, out):
        self.data.append(out)

    def _hook_impl(self):
        self.hook = self.module.register_forward_hook(lambda *args, **kwargs: self.forw(*args, **kwargs))

    def _unhook_impl(self):
        self.hook.remove()
        if self.do_rand:
            random.shuffle(self.data)
        self.result[self.layer_idx] = self.data


class UNetFFHookerIntervent(ObjectHooker[Attention]):
    def __init__(
        self,
        module,
        layer_idx,
        autoencoder,
        neurons_to_reset
    ):
        super().__init__(module)
        self.layer_idx = layer_idx
        self.autoencoder = autoencoder
        self.neurons_to_reset = neurons_to_reset

    def forw(self, module, inp, out):
        if self.neurons_to_reset is not None:
            hidden_acts = self.autoencoder.encode(out.to(dtype=torch.float32))
            hidden_acts[:, :, self.neurons_to_reset] = 0
            out = self.autoencoder.decode(hidden_acts)
        return out.to(dtype=torch.float16)


    def _hook_impl(self):
        self.hook = self.module.register_forward_hook(lambda *args, **kwargs: self.forw(*args, **kwargs))

    def _unhook_impl(self):
        self.hook.remove()
        

class UNetFFLocator(ModuleLocator[Attention]):
    def __init__(self, restrict: bool = None, locate_middle_block: bool = False):
        self.restrict = restrict
        self.layer_names = []

    def locate(self, model: UNet2DConditionModel) -> List[Attention]:
        """
        Locate all cross-attention modules in a UNet2DConditionModel.

        Args:
            model (`UNet2DConditionModel`): The model to locate the cross-attention modules in.

        Returns:
            `List[Attention]`: The list of cross-attention modules.
        """

        self.layer_names.clear()
        blocks_list = []
        up_names = ['up'] * len(model.up_blocks)
        down_names = ['down'] * len(model.down_blocks)

        counter = 0
        for unet_block, name in itertools.chain(
            zip(model.up_blocks, up_names),
            zip(model.down_blocks, down_names),
            zip([model.mid_block], ['mid'])
        ):
            if 'CrossAttn' in unet_block.__class__.__name__:
                blocks = []
                for spatial_transformer in unet_block.attentions:
                    for transformer_block in spatial_transformer.transformer_blocks:
                        blocks.append(transformer_block.ff)
                        
                blocks = [b for idx, b in enumerate(blocks)]
                names = [f'{name}-ff-{i}' for i in range(len(blocks))]
                blocks_list.extend(blocks)
                self.layer_names.extend(names)
        
        return blocks_list

trace = DiffusionFFHooker
trace_intervention = DiffusionFFHookerIntervent
