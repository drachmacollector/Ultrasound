"""Verify GradCAM target layer resolution for all backbones."""
import sys
sys.path.insert(0, ".")
import timm
from src.models.gradcam import get_target_layer, _BACKBONE_LAYER_PATHS

backbones = list(_BACKBONE_LAYER_PATHS.keys())

all_pass = True
for backbone_name in backbones:
    try:
        m = timm.create_model(backbone_name, pretrained=False)
        layer = get_target_layer(m, backbone_name)
        print(f"  {backbone_name}: {type(layer).__name__} -> OK")
    except Exception as e:
        print(f"  {backbone_name}: FAIL - {e}")
        all_pass = False

print()
if all_pass:
    print("All GradCAM target layers resolved successfully.")
else:
    print("Some target layers FAILED to resolve.")
    sys.exit(1)
