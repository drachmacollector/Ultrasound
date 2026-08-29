import os
import pandas as pd
from pathlib import Path

def build_manifest():
    input_csv = Path("data/raw/natalia_pbfus1/resume.csv")
    out_manifest = Path("data/processed/natalia_manifest.csv")
    out_excluded = Path("data/processed/natalia_excluded_no_taxonomy_match.csv")
    log_path = Path("logs/build_natalia_manifest_output.txt")
    
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_path, "w", encoding="utf-8") as f:
        def log(msg):
            print(msg)
            f.write(msg + "\n")
            
        log("--- Building NatalIA Manifest ---")
        df = pd.read_csv(input_csv)
        log(f"Loaded {len(df)} rows from {input_csv}")
        
        # Mapping logic
        # 0: Biparietal -> collapsed_label="Head"
        # 1: Abdominal -> canonical_label="Fetal_abdomen"
        # 2: Heart -> Excluded
        # 3: Spine -> Excluded
        # 4: Femur -> canonical_label="Fetal_femur"
        # 5: No Plane -> canonical_label="Other"
        
        included_rows = []
        excluded_rows = []
        
        for _, row in df.iterrows():
            val = row['value']
            row_dict = row.to_dict()
            
            if val == 2 or val == 3:
                excluded_rows.append(row_dict)
            else:
                if val == 0:
                    row_dict['collapsed_label'] = "Head"
                    row_dict['canonical_label'] = "" # No canonical label
                elif val == 1:
                    row_dict['canonical_label'] = "Fetal_abdomen"
                    row_dict['collapsed_label'] = ""
                elif val == 4:
                    row_dict['canonical_label'] = "Fetal_femur"
                    row_dict['collapsed_label'] = ""
                elif val == 5:
                    row_dict['canonical_label'] = "Other"
                    row_dict['collapsed_label'] = ""
                included_rows.append(row_dict)
                
        df_included = pd.DataFrame(included_rows)
        df_excluded = pd.DataFrame(excluded_rows)
        
        log(f"Excluded {len(df_excluded)} rows (Heart and Spine).")
        log(f"Included {len(df_included)} rows.")
        
        # Assertions
        assert len(df_included) == 19212, f"Expected 19212 rows, got {len(df_included)}"
        
        # Verify valid labels
        valid_canonical = {"Fetal_abdomen", "Fetal_femur", "Other", ""}
        valid_collapsed = {"Head", ""}
        
        invalid_can = df_included[~df_included['canonical_label'].isin(valid_canonical)]
        invalid_col = df_included[~df_included['collapsed_label'].isin(valid_collapsed)]
        
        assert len(invalid_can) == 0, f"Found invalid canonical labels: {invalid_can['canonical_label'].unique()}"
        assert len(invalid_col) == 0, f"Found invalid collapsed labels: {invalid_col['collapsed_label'].unique()}"
        
        df_included.to_csv(out_manifest, index=False)
        df_excluded.to_csv(out_excluded, index=False)
        
        log(f"Saved manifest to {out_manifest}")
        log(f"Saved excluded rows to {out_excluded}")
        log("Manifest build complete and verified.")

if __name__ == "__main__":
    build_manifest()
