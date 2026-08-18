import json
import pandas as pd
from pathlib import Path
import numpy as np

def derive_bbox(row, class_name, margin_ratio=0.2, img_width=None, img_height=None):
    xs = []
    ys = []
    
    if class_name == 'Head':
        cols = ['ofd_1_x', 'ofd_1_y', 'ofd_2_x', 'ofd_2_y', 'bpd_1_x', 'bpd_1_y', 'bpd_2_x', 'bpd_2_y']
    elif class_name == 'Abdomen':
        cols = ['tad_1_x', 'tad_1_y', 'tad_2_x', 'tad_2_y', 'apad_1_x', 'apad_1_y', 'apad_2_x', 'apad_2_y']
    elif class_name == 'Femur':
        cols = ['fl_1_x', 'fl_1_y', 'fl_2_x', 'fl_2_y']
    else:
        return None
        
    for i in range(0, len(cols), 2):
        if pd.notna(row[cols[i]]) and pd.notna(row[cols[i+1]]):
            try:
                xs.append(float(row[cols[i]]))
                ys.append(float(row[cols[i+1]]))
            except ValueError:
                pass
                
    if not xs or not ys:
        return None
        
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    width = x_max - x_min
    height = y_max - y_min
    
    # Use max of width/height for margin to handle thin structures (like femur)
    margin = margin_ratio * max(width, height)
    
    # Ensure minimum size to prevent 0-area boxes
    if margin == 0:
        margin = 10
        
    final_xmin = x_min - margin
    final_ymin = y_min - margin
    final_xmax = x_max + margin
    final_ymax = y_max + margin
    
    if img_width is not None and img_height is not None:
        final_xmin = max(0.0, min(final_xmin, img_width - 1.0))
        final_ymin = max(0.0, min(final_ymin, img_height - 1.0))
        final_xmax = max(0.0, min(final_xmax, img_width - 1.0))
        final_ymax = max(0.0, min(final_ymax, img_height - 1.0))
        
    return [final_xmin, final_ymin, final_xmax, final_ymax]


def main():
    base_dir = Path("data/raw/ucl_hc18/annotations")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    bboxes = {}
    
    # Map subset names to their internal labels for our multi-task head.
    # We will use 0: Head, 1: Abdomen, 2: Femur
    class_map = {'Head': 0, 'Abdomen': 1, 'Femur': 2}
    
    for ds_name in ['HC18', 'UCL']:
        ds_dir = base_dir / ds_name
        if not ds_dir.exists():
            continue
            
        for csv_file in ds_dir.glob("*.csv"):
            if "Test" in csv_file.name or "Train" in csv_file.name:
                continue # The main files usually contain all, let's process the main ones
                
            class_name = csv_file.stem
            if class_name not in class_map:
                continue
                
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                image_name = row['image_name']
                bbox = derive_bbox(row, class_name)
                if bbox is not None:
                    # In HC18, images might have prefix or just be the name. 
                    # We will store them by filename.
                    bboxes[image_name] = {
                        'bbox': bbox,
                        'class_id': class_map[class_name]
                    }

    with open(out_dir / "bboxes.json", "w", encoding="utf-8") as f:
        json.dump(bboxes, f, indent=2)
        
    print(f"Derived {len(bboxes)} bounding boxes. Saved to {out_dir / 'bboxes.json'}")

if __name__ == "__main__":
    main()
