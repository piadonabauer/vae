import cv2, torch
from pathlib import Path
import sys

def process_folder(folder_path):
    folder = Path(folder_path)

    if not folder.exists():
        raise RuntimeError(f"Folder does not exist: {folder}")

    participants = [f for f in folder.iterdir() if f.is_dir()]

    if not participants:
        raise RuntimeError(f"No participant folders found in {folder}")

    for participant in participants:
        expressions = [f for f in participant.iterdir() if f.is_dir()]

        for expr in expressions:
            output_file = expr / f"{expr.name}.pt"
            print(f"Processing: {expr}")

            if output_file.exists():
                print(f"Skipping {expr}, already processed.")
                continue

            mp4s = sorted(expr.glob("*.mp4"))

            if len(mp4s) < 2:
                print(f"Warning: Found only {len(mp4s)} mp4 files in {expr}, skipping...")
                continue

            videos = []

            for p in mp4s:
                print(f"  Processing video: {p.name}")

                cap = cv2.VideoCapture(str(p))
                frames = []

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    t = torch.from_numpy(frame).permute(2,0,1).float() / 255.0
                    frames.append(t)

                cap.release()

                if frames:
                    videos.append(torch.stack(frames, dim=1))  # C,T,H,W

            if videos:
                video_tensor = torch.stack(videos, dim=0)  # V,C,T,H,W
                print(f"  Shapes: {[v.shape for v in videos]} -> {video_tensor.shape}")

                output_file = expr / f"{expr.name}.pt"
                torch.save({"video": video_tensor}, output_file)

                print(f"  Saved: {output_file}")
            else:
                print(f"  No videos processed in {expr}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python save_multiview.py <folder_path>")
        sys.exit(1)

    process_folder(sys.argv[1])