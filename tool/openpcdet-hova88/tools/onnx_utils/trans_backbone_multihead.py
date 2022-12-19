import torch
import argparse
from torch import nn
import numpy as np
from onnx_backbone_2d import BaseBEVBackbone
from onnx_multihead import  AnchorHeadMulti
from pcdet.config import cfg, cfg_from_yaml_file

class backbone(nn.Module):
    def __init__(self, cfg , gridx , gridy, dense_head_input_channel=384):
        super().__init__()
        self.backbone_2d = BaseBEVBackbone(cfg.MODEL.BACKBONE_2D, cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES)
        self.dense_head =  AnchorHeadMulti(
            model_cfg=cfg.MODEL.DENSE_HEAD,
            input_channels=dense_head_input_channel,
            num_class=len(cfg.CLASS_NAMES),
            class_names=cfg.CLASS_NAMES,
            grid_size=np.array([gridx , gridy , 1]),
            point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
            predict_boxes_when_training=False)

    def forward(self, spatial_features):
        x = self.backbone_2d(spatial_features)
        batch_cls_preds, batch_box_preds = self.dense_head.forward(x)

        return batch_cls_preds, batch_box_preds

def build_backbone_multihead(ckpt , cfg ):
    assert cfg.MODEL.NAME == 'PointPillar'

    input_c = cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES
    checkpoint = torch.load(ckpt, map_location='cuda')

    pc_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
    voxel_size = np.array(cfg.DATA_CONFIG.DATA_PROCESSOR[2]['VOXEL_SIZE'])
    grid_size = (pc_range[3:] - pc_range[:3]) /voxel_size
    gridx = grid_size[0].astype(np.int)
    gridy = grid_size[1].astype(np.int)
    dense_head_input_channel = checkpoint["model_state"]['dense_head.shared_conv.0.weight'].shape[1]
    model = backbone(cfg , gridx ,gridy, dense_head_input_channel)
    model.to('cuda').eval()

    dicts = {}
    for key in checkpoint["model_state"].keys():
        if "backbone_2d" in key:
            dicts[key] = checkpoint["model_state"][key]
        if "dense_head" in key:
            dicts[key] = checkpoint["model_state"][key]
            print(key, checkpoint["model_state"][key].shape)
    print(dicts.keys())
    model.load_state_dict(dicts)

    dummy_input = torch.ones(1, input_c, gridx, gridy).cuda()
    return model , dummy_input

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--filename_mh', type=str, default=None, help='specify the config for training')
    parser.add_argument('--export_onnx_file', type=str, default=None, help='specify the config for training')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    import numpy as np 
    from pcdet.config import cfg, cfg_from_yaml_file
    args = parse_config()
    cfg_file = args.cfg_file #'/path/to/cbgs_pp_multihead.yaml'
    filename_mh = args.filename_mh #"/path/to/pp_multihead_nds5823_updated.pth"
    cfg_from_yaml_file(cfg_file, cfg)
    model , dummy_input = build_backbone_multihead(filename_mh , cfg )

    export_onnx_file = args.export_onnx_file #"/path/to/cbgs_pp_multihead_pfe.onnx"
    model.eval().cuda()
    torch.onnx.export(model,
                      dummy_input,
                      export_onnx_file,
                      opset_version=10,
                      verbose=True,
                      do_constant_folding=True) # 输出名
