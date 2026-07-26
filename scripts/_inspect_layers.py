"""Inspect named children for each backbone to identify correct GradCAM target layers."""
import sys
sys.path.insert(0, ".")
import timm
import torch.nn as nn

for name in ['repvgg_a1', 'mobilenetv3_large_100', 'efficientnet_lite0', 'tf_efficientnetv2_s']:
    m = timm.create_model(name, pretrained=False)
    print(f"--- {name} top-level ---")
    for n, c in m.named_children():
        print(f"  .{n}: {type(c).__name__}")

    # For blocks-based models, inspect last block group
    if hasattr(m, 'blocks'):
        blocks = m.blocks
        if isinstance(blocks, nn.Module):
            blocks_children = list(blocks.children())
            print(f"  .blocks has {len(blocks_children)} groups")
            last_group = blocks_children[-1]
            print(f"    last group type: {type(last_group).__name__}")
            if isinstance(last_group, nn.Module):
                last_module = list(last_group.children())[-1]
                print(f"    last module type: {type(last_module).__name__}")

    # For RepVGG, inspect stages
    if hasattr(m, 'stages'):
        stages = m.stages
        if isinstance(stages, nn.Module):
            stages_children = list(stages.children())
            print(f"  .stages has {len(stages_children)} stages")
            last_stage = stages_children[-1]
            print(f"    last stage type: {type(last_stage).__name__}")
            if hasattr(last_stage, '__iter__') or hasattr(last_stage, 'children'):
                if isinstance(last_stage, nn.Module):
                    last_stage_children = list(last_stage.children())
                    last_block = last_stage_children[-1] if last_stage_children else last_stage
                    print(f"    last block in last stage: {type(last_block).__name__}")
    print()
