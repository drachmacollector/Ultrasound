import torch
import torch.nn as nn
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
import timm
from collections import OrderedDict

class ConvNeXtBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = timm.create_model('convnext_tiny', pretrained=True, features_only=True, out_indices=[1, 2, 3])
        
    def forward(self, x):
        features = self.body(x)
        out = OrderedDict()
        out['0'] = features[0]
        out['1'] = features[1]
        out['2'] = features[2]
        return out

class DetBackbone(nn.Module):
    def __init__(self, body, fpn, cls_head):
        super().__init__()
        self.body = body
        self.fpn = fpn
        self.cls_head = cls_head
        self.out_channels = 256
        self.cls_logits = None
        
    def forward(self, x):
        features = self.body(x)
        self.cls_logits = self.cls_head(features['2'])
        fpn_features = self.fpn(features)
        return fpn_features


class MultiTaskConvNeXt(nn.Module):
    def __init__(self, num_cls_classes=8, num_det_classes=4):
        super().__init__()
        self.convnext = ConvNeXtBackbone()
        
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.LayerNorm(768),
            nn.Linear(768, num_cls_classes)
        )
        
        from torchvision.ops import feature_pyramid_network
        self.fpn = feature_pyramid_network.FeaturePyramidNetwork(
            in_channels_list=[192, 384, 768],
            out_channels=256
        )
        
        self.det_backbone = DetBackbone(self.convnext, self.fpn, self.cls_head)
        
        anchor_sizes = ((32, 64, 128), (64, 128, 256), (128, 256, 512))
        aspect_ratios = ((0.5, 1.0, 2.0),) * 3
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        
        self.retinanet = RetinaNet(
            backbone=self.det_backbone,
            num_classes=num_det_classes,
            anchor_generator=anchor_generator,
            # We skip torchvision's normalization because our albumentations pipeline already does it
            image_mean=[0.0, 0.0, 0.0],
            image_std=[1.0, 1.0, 1.0],
            min_size=224,
            max_size=224
        )

    def forward(self, images, targets=None):
        """
        images: list of Tensors or Tensor of shape (B, 3, H, W)
        targets: list of dicts for detection
        Returns:
            cls_logits: (B, num_cls_classes)
            det_output: dict of losses (if training and targets present) or list of detections (if eval)
        """
        if isinstance(images, torch.Tensor):
            images = list(image for image in images)
            
        det_output = self.retinanet(images, targets)
        cls_logits = self.det_backbone.cls_logits
        
        return cls_logits, det_output
