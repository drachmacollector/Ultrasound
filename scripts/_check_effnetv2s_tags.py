"""Check tf_efficientnetv2_s pretrained tags and their pretrained_cfg."""
import sys
sys.path.insert(0, ".")
import timm

tags_to_check = [
    "tf_efficientnetv2_s",
    "tf_efficientnetv2_s.in1k",
    "tf_efficientnetv2_s.in21k",
    "tf_efficientnetv2_s.in21k_ft_in1k",
]

print("=== tf_efficientnetv2_s pretrained tag comparison ===\n")
for tag in tags_to_check:
    try:
        m = timm.create_model(tag, pretrained=False)
        cfg: dict = getattr(m, "pretrained_cfg", {})
        print(f"Tag: {tag!r}")
        print(f"  input_size : {cfg.get('input_size')}")
        print(f"  test_input_size: {cfg.get('test_input_size', 'N/A')}")
        print(f"  mean       : {cfg.get('mean')}")
        print(f"  std        : {cfg.get('std')}")
        print(f"  crop_pct   : {cfg.get('crop_pct')}")
        print(f"  crop_mode  : {cfg.get('crop_mode', 'N/A')}")
        print(f"  num_classes: {cfg.get('num_classes', 'N/A')}")
        print()
    except Exception as e:
        print(f"Tag: {tag!r}  -> ERROR: {e}\n")
