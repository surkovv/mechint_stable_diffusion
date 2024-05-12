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

from .utils import cache_dir, auto_autocast
from .hook import ObjectHooker, AggregateHooker, UNetCrossAttentionLocator, ModuleLocator


__all__ = ['trace', 'DiffusionFFHooker']


class DiffusionFFHooker(AggregateHooker):
    def __init__(
            self,
            pipeline: Union[StableDiffusionPipeline, StableDiffusionXLPipeline],
            result: dict,
            do_rand: bool,
            restrict: list = [7]
    ):
        self.locator = UNetFFLocator()

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
        to_save = torch.cat(self.data, dim=0).flatten(0, 1)
        if self.do_rand:
            idx = torch.randperm(to_save.shape[0])
            to_save = to_save[idx, :]
        self.result[self.layer_idx] = to_save


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
            zip([model.mid_block], ['mid']),
            zip(model.down_blocks, down_names)
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
