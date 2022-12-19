import torch
import argparse
from torch import nn
import numpy as np
from onnx_backbone_2d import BaseBEVBackbone
from onnx_centerhead import CenterHead
from pcdet.config import cfg, cfg_from_yaml_file

class backbone(nn.Module):
    def __init__(self, cfg, voxel_size, gridx, gridy):
        super().__init__()
        self.backbone_2d = BaseBEVBackbone(cfg.MODEL.BACKBONE_2D, cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES)
        self.dense_head = CenterHead(
            model_cfg=cfg.MODEL.DENSE_HEAD,
            input_channels=384,
            num_class=len(cfg.CLASS_NAMES),
            class_names=cfg.CLASS_NAMES,
            grid_size=np.array([gridx , gridy , 1]),
            point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
            voxel_size=voxel_size,
            predict_boxes_when_training=False)

    def forward(self, spatial_features):
        x = self.backbone_2d(spatial_features)
        pred_dicts = self.dense_head.forward(x)

        batch_cls_preds = []
        batch_box_preds = []
        for pred_dict in pred_dicts:
            batch_cls_preds.append(pred_dict['pred_scores'])
            batch_box_preds.append(pred_dict['pred_boxes'])

        return batch_cls_preds, batch_box_preds

def build_backbone_centernet(ckpt, cfg):
    assert cfg.MODEL.NAME == 'CenterPoint'
    input_c = cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES

    pc_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
    voxel_size = np.array(cfg.DATA_CONFIG.DATA_PROCESSOR[2]['VOXEL_SIZE'])
    grid_size = (pc_range[3:] - pc_range[:3]) / voxel_size
    gridx = grid_size[0].astype(np.int)
    gridy = grid_size[1].astype(np.int)
    model = backbone(cfg, voxel_size, gridx, gridy)
    model.to('cuda').eval()

    checkpoint = torch.load(ckpt, map_location='cuda')
    dicts = {}
    for key in checkpoint["model_state"].keys():
        if "backbone_2d" in key:
            dicts[key] = checkpoint["model_state"][key]
        if "dense_head" in key:
            dicts[key] = checkpoint["model_state"][key]
    model.load_state_dict(dicts)

    # for name, paramer in model.named_parameters():
    #     if 'shared_conv' in name:
    #         print(name, paramer)

    # dummy_input = torch.ones(1, input_c, gridx, gridy).cuda()
    # dummy_input = torch.rand(1, input_c, gridx, gridy).cuda()
    dummy_input = torch.load("/home/daodao/mytensor.pt")
    return model, dummy_input

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='')
    parser.add_argument('--ckpt', type=str, default=None, help='')
    parser.add_argument('--onnx_name', type=str, default=None, help='')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    import numpy as np 
    from pcdet.config import cfg, cfg_from_yaml_file
    args = parse_config()
    cfg_file = args.cfg_file
    filename_mh = args.ckpt
    cfg_from_yaml_file(cfg_file, cfg)
    model, dummy_input = build_backbone_centernet(filename_mh , cfg )

    export_onnx_file = args.onnx_name 
    model.eval().cuda()
    torch.onnx.export(model,
                      dummy_input,
                      export_onnx_file,
                      opset_version=10,
                      verbose=True,
                      do_constant_folding=True)
