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

