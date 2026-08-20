import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
from src.models.multitask_model import MultiTaskConvNeXt

model = MultiTaskConvNeXt(pretrained=False)
model.train()

images = [torch.rand(3, 224, 224)]
# Empty targets
targets = [{"boxes": torch.empty((0, 4), dtype=torch.float32), "labels": torch.empty((0,), dtype=torch.int64)}]

cls_logits, det_losses = model(images, targets)
print("det_losses:", det_losses)

if isinstance(det_losses, dict):
    for k, v in det_losses.items():
        print(f"{k}: {v} (type: {type(v)})")
    
    det_loss = sum(loss for loss in det_losses.values())
    print("sum:", det_loss, "type:", type(det_loss))
