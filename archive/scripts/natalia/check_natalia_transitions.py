import pandas as pd
from pathlib import Path

def check_transitions():
    csv_path = Path("data/raw/natalia_pbfus1/resume.csv")
    df = pd.read_csv(csv_path)
    
    # Exclude heart/spine
    df = df[~df['value'].isin([2, 3])]

    # value mappings: 0=Head, 1=Abdomen, 4=Femur, 5=Other
    valid_values = {0, 1, 4}

    transitions = 0
    
    for studie, group in df.groupby('studie'):
        # sort by cineframe index
        def get_idx(fn):
            try:
                # cineframe_NNN_...
                return int(fn.split('_')[1])
            except:
                return 0
                
        group = group.copy()
        group['idx'] = group['file_name'].apply(get_idx)
        group = group.sort_values('idx')
        
        last_plane = None
        for val in group['value']:
            if val in valid_values:
                if last_plane is not None and last_plane != val:
                    transitions += 1
                last_plane = val

    print(f'Found {transitions} standard-plane to different-standard-plane transitions')

    log_path = Path("logs/natalia_transition_investigation.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("--- Step A: Transition Investigation ---\n")
        f.write(f"Found {transitions} genuine standard-plane to different-standard-plane transitions in continuous sequences.\n")
        if transitions < 5:
            f.write("Decision Gate: Statistically unusable number of transitions (<5).\n")
            f.write("Do not force a latency number. This limitation will be documented in EVAL_REPORT.md exactly as was done for IUGC.\n")
        else:
            f.write("Decision Gate: Sufficient transitions found. (Task 2.2 Step C should proceed if verified visually).\n")

if __name__ == "__main__":
    check_transitions()
