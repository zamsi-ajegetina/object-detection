import os
import cv2
import argparse
from pathlib import Path

def extract_frames(video_dir, output_dir, target_fps=3):
    """
    Extract frames from videos in video_dir at a target FPS and save them to output_dir,
    categorised by environment based on the filename.
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    
    # Ensure base output directories exist
    envs = ['highway', 'residential', 'market', 'unclassified']
    for env in envs:
        (output_dir / env).mkdir(parents=True, exist_ok=True)

    valid_extensions = {'.mp4', '.mov', '.avi', '.mkv'}
    video_files = [f for f in video_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]

    if not video_files:
        print(f"No videos found in {video_dir}")
        return

    total_extracted = 0

    for video_path in video_files:
        filename = video_path.name.lower()
        
        # Determine environment from filename
        env_cat = 'unclassified'
        if 'highway' in filename:
            env_cat = 'highway'
        elif 'residential' in filename:
            env_cat = 'residential'
        elif 'market' in filename or 'urban' in filename:
            env_cat = 'market'
            
        env_out_dir = output_dir / env_cat
        video_name_stem = video_path.stem

        print(f"Processing {video_path.name} -> {env_cat} (Target FPS: {target_fps})")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  Failed to open {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps == 0 or total_frames == 0:
            print(f"  Invalid video properties (FPS: {fps}, Total Frames: {total_frames})")
            continue
            
        # Calculate frame skip interval
        if target_fps >= fps:
            frame_skip = 1
        else:
            frame_skip = int(fps / target_fps)
            
        print(f"  Original FPS: {fps:.2f}, Target: {target_fps}, Extracting 1 every {frame_skip} frames")

        count = 0
        extracted = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % frame_skip == 0:
                frame_name = f"{video_name_stem}_frame_{count:06d}.jpg"
                out_path = env_out_dir / frame_name
                # cv2.imwrite(str(out_path), frame)
                # We'll also resize to something manageable like 720p to save space.
                height, width = frame.shape[:2]
                if height > 720:
                    scale = 720 / height
                    new_width = int(width * scale)
                    frame = cv2.resize(frame, (new_width, 720))
                
                cv2.imwrite(str(out_path), frame)
                extracted += 1
                
            count += 1
            
        cap.release()
        total_extracted += extracted
        print(f"  Extracted {extracted} frames from {video_path.name}")
        
    print(f"\nTotal frames extracted across all videos: {total_extracted}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos for PROSIT 2 dataset")
    parser.add_argument("--video_dir", type=str, default="videos", help="Directory containing input videos")
    parser.add_argument("--output_dir", type=str, default="data/raw/ghana", help="Directory to save extracted frames")
    parser.add_argument("--fps", type=int, default=3, help="Target frames per second to extract")
    args = parser.parse_args()

    extract_frames(args.video_dir, args.output_dir, args.fps)
