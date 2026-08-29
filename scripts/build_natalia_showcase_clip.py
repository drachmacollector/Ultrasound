import argparse
import logging
from pathlib import Path
import cv2
import pandas as pd
import imageio

def build_clip(exam_name: str, out_path: Path):
    log = logging.getLogger(__name__)
    
    # Check that directory exists
    data_dir = Path("data/raw/natalia_pbfus1")
    exam_dir = data_dir / exam_name
    if not exam_dir.exists():
        log.error(f"Directory {exam_dir} does not exist.")
        return
        
    csv_path = data_dir / "resume.csv"
    if not csv_path.exists():
        log.error(f"Manifest {csv_path} does not exist.")
        return
        
    df = pd.read_csv(csv_path)
    df_exam = df[df['studie'] == exam_name].copy()
    
    if len(df_exam) == 0:
        log.error(f"No frames found for exam {exam_name} in resume.csv.")
        return
        
    # Sort frames by cineframe index
    def get_idx(fn):
        try:
            return int(fn.split('_')[1])
        except:
            return 0
            
    df_exam['idx'] = df_exam['file_name'].apply(get_idx)
    df_exam = df_exam.sort_values('idx')
    
    log.info(f"Found {len(df_exam)} frames for {exam_name}. Compiling into {out_path}...")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    writer = imageio.get_writer(
        str(out_path),
        fps=24,
        macro_block_size=None,
        pixelformat="yuv420p",
        codec="libx264"
    )
    
    frames_written = 0
    for _, row in df_exam.iterrows():
        img_path = exam_dir / row['file_name']
        if not img_path.exists():
            log.warning(f"File {img_path} missing, skipping.")
            continue
            
        # read image
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
            
        # BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        frames_written += 1
        
    writer.close()
    log.info(f"Successfully wrote {frames_written} frames to {out_path}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exam", required=True, help="Exam folder name (e.g. 'Obstetrics Exam - 02-May-2024_1144_AM')")
    parser.add_argument("--out", required=True, help="Output MP4 path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    
    build_clip(args.exam, Path(args.out))

if __name__ == "__main__":
    main()
