import torch
import torch.nn as nn
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
import timm
from collections import OrderedDict

class ConvNeXtBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.body = timm.create_model('convnext_tiny.fb_in22k_ft_in1k', pretrained=pretrained, features_only=True, out_indices=[1, 2, 3])
        
    def forward(self, x):
        features = self.body(x)
        out = OrderedDict()
        out['0'] = features[0]
        out['1'] = features[1]
        out['2'] = features[2]
        return out


class MultiTaskConvNeXt(nn.Module):
    def __init__(self, num_cls_classes=8, num_det_classes=3, pretrained=True):
        super().__init__()
        self.convnext = ConvNeXtBackbone(pretrained=pretrained)
        
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
        
        # Dummy backbone to satisfy RetinaNet's init
        class DummyBackbone(nn.Module):
            out_channels = 256
            def forward(self, x): return x
            
        anchor_sizes = ((32, 64, 128), (64, 128, 256), (128, 256, 512))
        aspect_ratios = ((0.5, 1.0, 2.0),) * 3
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        
        self.retinanet = RetinaNet(
            backbone=DummyBackbone(),
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
            
        original_image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        
        # 1. Transform images and targets using RetinaNet's built-in transform
        images_transformed, targets_transformed = self.retinanet.transform(images, targets)
        
        # 2. Run backbone once
        features = self.convnext(images_transformed.tensors)
        
        # 3. Compute classification logits
        cls_logits = self.cls_head(features['2'])
        
        # 4. Compute FPN features
        fpn_features = self.fpn(features)
        features_list = list(fpn_features.values())
        
        # 5. Run RetinaNet detection head
        head_outputs = self.retinanet.head(features_list)
        anchors = self.retinanet.anchor_generator(images_transformed, features_list)
        
        if self.training:
            assert targets_transformed is not None
            det_output = self.retinanet.compute_loss(targets_transformed, head_outputs, anchors)
        else:
            detections = self.retinanet.postprocess_detections(head_outputs, anchors, images_transformed.image_sizes)
            det_output = self.retinanet.transform.postprocess(detections, images_transformed.image_sizes, original_image_sizes)
            
        return cls_logits, det_output
