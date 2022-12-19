import torch
import torch.nn as nn

class PassThrough(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict, **kwargs):
        return batch_dict
