# IUGC2024 Dataset Documentation

## Dataset Structure

```
new/
├── train/                    # Training set
│   ├── videos/              # Video files directory
│   ├── seg/                 # Segmentation annotations
│   │   ├── seg_info.csv     # Segmentation annotation metadata
│   │   ├── landmark.json    # Landmark coordinates and parameters
│   │   └── [video_name]/    # Per-video segmentation folder
│   │       └── mask/        # Segmentation mask images
│   ├── cls/                 # Classification annotations
│   │   └── class_label.csv  # Classification labels
│   └── train_info.csv       # Training set metadata
├── val/                     # Validation set
│   ├── videos/              # Video files directory
│   ├── seg/                 # Segmentation annotations
│   │   ├── seg_info.csv     # Segmentation annotation metadata
│   │   ├── landmark.json    # Landmark coordinates and parameters
│   │   └── *.png            # Segmentation mask images
│   ├── cls/                 # Classification annotations
│   │   └── cls_label.csv    # Classification labels
│   └── val_info.csv         # Validation set metadata
└── test/                    # Test set
    ├── videos/              # Video files directory
    ├── seg/                 # Segmentation annotations
    │   ├── seg_info.csv     # Segmentation annotation metadata
    │   └── *.png            # Segmentation mask images
    ├── cls/                 # Classification annotations
    │   └── cls_label.csv    # Classification labels
    └── test_info.csv        # Test set metadata
```

## CSV File Formats

### train_info.csv
Contains the following columns:
- `filename`: Video filename
- `pos`: Whether the video is a standard plane (TRUE=standard plane, FALSE=non-standard plane)
- `frame_count`: Total number of frames in the video
- `labeled_frame_count`: Number of labeled frames
- `labeled_frame_index`: Index of labeled frames (comma-separated string)

**Example**:
```csv
filename,pos,frame_count,labeled_frame_count,labeled_frame_index
20190909T155747I1.avi,TRUE,80,9,"0,19,29,39,49,59,69,79,9"
20190726T095643_0.avi,FALSE,421,null,null
```

### val_info.csv / test_info.csv
Contains the following columns:
- `filename`: Video filename
- `SP_count`: Number of SP (standard plane) samples
- `NSP_count`: Number of NSP (non-standard plane) samples
- `frame_count`: Total number of frames in the video
- `labeled_frame_count`: Number of labeled frames (usually 1)
- `labeled_frame_index`: Index of the labeled frame
- `SP_index`: Frame index list of SP samples
- `NSP_index`: Frame index list of NSP samples

**Example**:
```csv
filename,SP_count,NSP_count,frame_count,labeled_frame_count,labeled_frame_index,SP_index,NSP_index
test_10_80.avi,72,9,81,1,80,"[9, 10, 11, ..., 80]","[0, 1, 2, ..., 8]"
```

### train/seg/seg_info.csv
Contains the following columns:
- `filename`: Video filename
- `frame_count`: Total number of frames in the video
- `labeled_frame_count`: Number of labeled frames
- `labeled_index`: Index of labeled frames (comma-separated string)

**Example**:
```csv
filename,frame_count,labeled_frame_count,labeled_index
20190909T155747I1.avi,80,9,"0,19,29,39,49,59,69,79,9"
```

### val/seg/seg_info.csv / test/seg/seg_info.csv
Contains the following columns:
- `filename`: Video filename
- `frame_count`: Total number of frames in the video
- `labeled_frame_count`: Number of labeled frames (usually 1)
- `labeled_frame_index`: Index of the labeled frame

**Example**:
```csv
filename,frame_count,labeled_frame_count,labeled_frame_index
202012132137152159400I1.avi,20,1,6
```

### train/cls/class_label.csv
Contains the following columns:
- `filename`: Video filename
- `frame_count`: Total number of frames in the video
- `pos_index`: Standard plane frame index ("NONE" means no standard plane, "ALL" means all frames are standard planes, or a specific index list)
- `neg_index`: Non-standard plane frame index ("ALL" means all frames are non-standard planes, or a specific index list)

**Example**:
```csv
filename,frame_count,pos_index,neg_index
20190726T095643_0.avi,421,NONE,ALL
```

### val/cls/cls_label.csv / test/cls/cls_label.csv
- Same format as `train/cls/class_label.csv`.

## Landmark JSON Format

### train/seg/landmark.json / val/seg/landmark.json / test/seg/landmark.json
Contains landmark coordinates and measurement parameters for each segmentation mask image.

**Format**:
- The JSON file is a dictionary where keys are mask image filenames
- Each value is an object containing:
  - `ps_points`: Array of two PS landmark points, each point is `[y, x]` format (strings)
  - `hsd_point`: HSD landmark point in `[y, x]` format (strings)
  - `aop_tangency`: AOP (angle of progression) tangency point in `[y, x]` format (strings)
  - `hsd`: HSD parameter value (float)
  - `aop`: AOP parameter value (float, in degrees)

**Important Notes**:
- Coordinates are given in **y-first, x-second** order (i.e., `[y, x]`)
- The coordinate origin is at the **top-left corner** of the image
- All coordinate values are stored as strings in the JSON file

**Example**:
```json
{
  "20190909T155747I1_0_6.png": {
    "ps_points": [
      ["69", "198"],
      ["84", "299"]
    ],
    "hsd_point": ["147", "294"],
    "aop_tangency": ["174", "339"],
    "hsd": 63.198101237299845,
    "aop": 122.41001622248665
  }
}
```

## File Correspondence

### 1. Correspondence between Video Files and Segmentation Annotations

#### Training Set
- **Video file**: `train/videos/20190909T155747I1.avi`
- **Segmentation metadata**: Matching row in `train/seg/seg_info.csv` where `filename` column matches
- **Segmentation mask images**: 
  - According to the `labeled_index` field in `seg_info.csv` (e.g., `"0,19,29,39,49,59,69,79,9"`), each frame corresponds to one mask image
  - Mask image path: `train/seg/[video_name]/mask/[video_name]_[frame]_6.png`
  - Example: `train/seg/20190909T155747I1/mask/20190909T155747I1_0_6.png` corresponds to frame 0
  - Example: `train/seg/20190909T155747I1/mask/20190909T155747I1_19_6.png` corresponds to frame 19
- **Landmark data**: 
  - Landmark coordinates and parameters are stored in `train/seg/landmark.json`
  - The key in the JSON file is the mask image filename (e.g., `"20190909T155747I1_0_6.png"`)
  - Each entry contains PS points, HSD point, AOP tangency point, and calculated HSD/AOP values

#### Validation Set / Test Set
- **Video file**: `val/videos/202012132137152159400I1.avi` or `test/videos/test_10_80.avi`
- **Segmentation metadata**: Matching row in `val/seg/seg_info.csv` or `test/seg/seg_info.csv` where `filename` column matches
- **Segmentation mask images**: 
  - According to the `labeled_frame_index` field in `seg_info.csv` (e.g., `6`), there is only one mask image
  - Mask image path: validation set uses `val/seg/[video_name]_[frame].png`, test set uses `test/seg/[video_name].png` (video name itself contains frame index information)
  - Example: `val/seg/202012132137152159400I1_6.png` corresponds to frame 6
  - Example: `test/seg/test_10_80.png` corresponds to frame 80 of `test_10_80.avi`
- **Landmark data**: 
  - Landmark coordinates and parameters are stored in `val/seg/landmark.json` or `test/seg/landmark.json`
  - The key in the JSON file is the mask image filename (e.g., `"202012132137152159400I1_6.png"` or `"test_10_80.png"`)
  - Each entry contains PS points, HSD point, AOP tangency point, and calculated HSD/AOP values
