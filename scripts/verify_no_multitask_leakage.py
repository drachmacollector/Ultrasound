import pandas as pd
from pathlib import Path
import sys

def main():
    train_path = Path("data/splits/multitask_train.csv")
    val_path = Path("data/splits/multitask_val.csv")
    
    if not train_path.exists() or not val_path.exists():
        print("Manifests not found. Run scripts/build_multitask_manifest.py first.")
        sys.exit(1)
        
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    
    # 1. Verify no FP or MULTICENTRE data
    for df, name in [(df_train, "Train"), (df_val, "Val")]:
        has_fp = df['image_path'].str.contains('/FP/').any()
        has_mc = df['image_path'].str.contains('/MULTICENTRE/').any()
        
        assert not has_fp, f"{name} set contains FP data!"
        assert not has_mc, f"{name} set contains MULTICENTRE data!"
        
    # 2. Verify patient disjointness
    train_patients = set(df_train[df_train['patient_id'] != -1]['patient_id'].unique())
    val_patients = set(df_val[df_val['patient_id'] != -1]['patient_id'].unique())
    
    overlap = train_patients.intersection(val_patients)
    assert len(overlap) == 0, f"Patient leakage detected! {len(overlap)} patients overlap between train and val: {overlap}"
    
    print(f"Verified! 0 FP/MULTICENTRE images found. 0 overlapping patients.")
    
if __name__ == "__main__":
    main()
