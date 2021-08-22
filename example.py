import torch
import torch.nn as nn
from video_swin_transformer import SwinTransformer3D


'''
initialize a SwinTransformer3D model
'''
model = SwinTransformer3D()
print(model)

dummy_x = torch.rand(1, 3, 32, 224, 224)
logits = model(dummy_x)
print(logits.shape)

'''
load the pretrained weight

1. git clone https://github.com/SwinTransformer/Video-Swin-Transformer.git
2. move all files into ./Video-Swin-Transformer

'''
# from mmcv import Config, DictAction
# from mmaction.models import build_model
# from mmcv.runner import get_dist_info, init_dist, load_checkpoint

# config = './configs/recognition/swin/swin_base_patch244_window1677_sthv2.py'
# checkpoint = './checkpoints/swin_base_patch244_window1677_sthv2.pth'

# cfg = Config.fromfile(config)
# model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
# load_checkpoint(model, checkpoint, map_location='cpu')

'''
use the pretrained SwinTransformer3D as feature extractor
'''

# [batch_size, channel, temporal_dim, height, width]
dummy_x = torch.rand(1, 3, 32, 224, 224)

# SwinTransformer3D without cls_head
backbone = model.backbone

# [batch_size, hidden_dim, temporal_dim/2, height/32, width/32]
feat = backbone(dummy_x)

# alternative way
feat = model.extract_feat(dummy_x)

# mean pooling
feat = feat.mean(dim=[2,3,4]) # [batch_size, hidden_dim]

# project
batch_size, hidden_dim = feat.shape
feat_dim = 512
proj = nn.Parameter(torch.randn(hidden_dim, feat_dim))

# final output
output = feat @ proj # [batch_size, feat_dim]
