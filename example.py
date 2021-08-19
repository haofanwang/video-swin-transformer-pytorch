import torch
from video_swin_transformer import SwinTransformer3D

model = SwinTransformer3D()
print(model)

dummy_x = torch.rand(1, 3, 32, 224, 224)
logits = model(dummy_x)
print(logits.shape)


from mmcv import Config, DictAction
from mmaction.models import build_model
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

config = './configs/recognition/swin/swin_base_patch244_window1677_sthv2.py'
checkpoint = 'swin_base_patch244_window1677_sthv2.pth'

cfg = Config.fromfile(config)
model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, checkpoint, map_location='cpu')

# Video Swin Transformer
backbone = model.backbone

# [batch_size, channel, temporal_dim, height, width]
dummy_x = torch.rand(1, 3, 32, 224, 224)

# [batch_size, new_channel, temporal_dim/2, height/32, width/32]
output = backbone(dummy_x)