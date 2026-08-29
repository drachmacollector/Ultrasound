import os
import pandas as pd
from pathlib import Path

def verify_structural():
    data_dir = Path("data/raw/natalia_pbfus1")
    log_path = Path("logs/natalia_structural_verification.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_path, "w", encoding="utf-8") as f:
        def log(msg):
            print(msg)
            f.write(msg + "\n")
            
        log("--- NatalIA Structural Verification ---")
        
        # 1. Assert exactly 90 subdirectories matching the pattern Obstetrics Exam - *
        subdirs = [d.name for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("Obstetrics Exam - ")]
        log(f"Found {len(subdirs)} 'Obstetrics Exam - *' directories.")
        assert len(subdirs) == 90, f"Expected 90 subdirectories, found {len(subdirs)}"
        
        # 2. Assert resume.csv exists with exactly 19,407 rows and correct columns
        csv_path = data_dir / "resume.csv"
        assert csv_path.exists(), "resume.csv not found"
        df = pd.read_csv(csv_path)
        log(f"resume.csv has {len(df)} rows.")
        assert len(df) == 19407, f"Expected 19407 rows in resume.csv, found {len(df)}"
        expected_cols = {"file_name", "studie", "class", "value"}
        assert expected_cols.issubset(set(df.columns)), f"Missing columns in resume.csv. Found: {df.columns}"
        
        # 3. Assert value column per-class counts
        val_counts = df["value"].value_counts().to_dict()
        expected_counts = {
            0: 42,
            1: 63,
            2: 61,
            3: 134,
            4: 46,
            5: 19061
        }
        log("Class counts:")
        for v in sorted(val_counts.keys()):
            log(f"  Class {v}: {val_counts[v]}")
            
        for k, v in expected_counts.items():
            assert val_counts.get(k, 0) == v, f"Expected {v} for class {k}, got {val_counts.get(k, 0)}"
        assert set(val_counts.keys()).issubset(set(expected_counts.keys())), "Unexpected class values found"
        
        # 4. Check for metadata.csv
        meta_path = data_dir / "metadata.csv"
        if meta_path.exists():
            meta_df = pd.read_csv(meta_path)
            log(f"metadata.csv found. Columns: {list(meta_df.columns)}")
        else:
            log("metadata.csv not found.")
            
        # 5. Assert every file_name resolves to an actual .jpeg inside its studie subdirectory
        log("Checking file existence for all 19,407 rows...")
        missing_files = []
        for i, row in df.iterrows():
            studie = row['studie']
            fname = row['file_name']
            fpath = data_dir / studie / fname
            if not fpath.exists():
                missing_files.append(fpath)
                
        if missing_files:
            log(f"WARNING: {len(missing_files)} files missing. Examples: {missing_files[:5]}")
        else:
            log("SUCCESS: All 19,407 files exist.")
            
        assert len(missing_files) == 0, f"{len(missing_files)} files missing!"
        
        log("Verification complete. All assertions passed.")

if __name__ == "__main__":
    verify_structural()
