"""
Temporary import verification script — checks all Phase 4 modules import cleanly.
"""
import sys
import importlib.util
sys.path.insert(0, ".")

errors = []

# Backbone
try:
    from src.models.backbone import build_model, get_pretrained_cfg
    print("src.models.backbone: OK")
except Exception as e:
    errors.append(f"src.models.backbone: FAIL - {e}")
    print(f"src.models.backbone: FAIL - {e}")

# Gradcam
try:
    from src.models.gradcam import get_target_layer, run_gradcam, _BACKBONE_LAYER_PATHS
    print("src.models.gradcam: OK")
except Exception as e:
    errors.append(f"src.models.gradcam: FAIL - {e}")
    print(f"src.models.gradcam: FAIL - {e}")

# Train
try:
    from src.train.train import load_config, build_class_weight_tensor, train
    print("src.train.train: OK")
except Exception as e:
    errors.append(f"src.train.train: FAIL - {e}")
    print(f"src.train.train: FAIL - {e}")

# LR finder
try:
    spec = importlib.util.spec_from_file_location("lr_finder", "scripts/lr_finder.py")
    if spec is None or spec.loader is None:
        raise ImportError("Could not find spec or loader for lr_finder")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("scripts.lr_finder: OK")
except Exception as e:
    errors.append(f"scripts.lr_finder: FAIL - {e}")
    print(f"scripts.lr_finder: FAIL - {e}")

# Smoke test
try:
    spec2 = importlib.util.spec_from_file_location("smoke_test", "scripts/smoke_test.py")
    if spec2 is None or spec2.loader is None:
        raise ImportError("Could not find spec or loader for smoke_test")
    mod2 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(mod2)
    print("scripts.smoke_test: OK")
except Exception as e:
    errors.append(f"scripts.smoke_test: FAIL - {e}")
    print(f"scripts.smoke_test: FAIL - {e}")

print()
if errors:
    print("IMPORT ERRORS:")
    for err in errors:
        print(" ", err)
    sys.exit(1)
else:
    print("All imports OK.")
