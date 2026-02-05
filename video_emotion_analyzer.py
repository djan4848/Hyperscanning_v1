import os
import argparse

# ---- Environment tweaks (before importing DeepFace / TF) ----
# Force CPU (optional; keep if you want reproducibility / avoid GPU issues)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Suppress TensorFlow logs for cleaner output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import pandas as pd
from tqdm import tqdm
from deepface import DeepFace


class VideoEmotionAnalyzer:
    def __init__(self, detector_backend: str = "opencv", enforce_detection: bool = False):
        """
        Args:
            detector_backend: Face detector backend ('opencv','ssd','dlib','mtcnn','retinaface','mediapipe',...)
            enforce_detection: If True, DeepFace raises when no face is found in a frame.
        """
        self.detector_backend = detector_backend
        self.enforce_detection = enforce_detection
        self.emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def analyze_frame(self, frame):
        """
        Returns:
            dict of emotion scores (keys in self.emotions) or None if no face / error.
        """
        try:
            result = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                detector_backend=self.detector_backend,
                enforce_detection=self.enforce_detection,
                silent=True,
            )

            # DeepFace may return a list (one dict per face). We take the first.
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("emotion", None)

            # Or it may return a dict (depending on version/settings)
            if isinstance(result, dict):
                return result.get("emotion", None)

            return None

        except ValueError:
            return None
        except Exception:
            return None

    def process_video(self, video_path, output_csv_path=None, process_every_n_frames=5):
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if not original_fps or original_fps <= 0:
            original_fps = 30.0  # fallback

        print(f"Processing {os.path.basename(video_path)}...")
        print(
            f"Total Frames: {total_frames}, FPS: {original_fps:.2f}, "
            f"Process every {process_every_n_frames} frames, Backend: {self.detector_backend}"
        )

        results = []
        frame_idx = 0

        pbar = tqdm(total=total_frames, unit="frame")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % process_every_n_frames == 0:
                timestamp = frame_idx / original_fps
                emotion_data = self.analyze_frame(frame)

                row = {
                    "frame_id": frame_idx,
                    "timestamp_seconds": round(timestamp, 4),
                    "face_detected": False,
                }

                if isinstance(emotion_data, dict) and len(emotion_data) > 0:
                    row["face_detected"] = True
                    for emo in self.emotions:
                        row[f"prob_{emo}"] = float(emotion_data.get(emo, 0.0))
                else:
                    # Keep NaN when no face; avoids injecting artificial zeros
                    for emo in self.emotions:
                        row[f"prob_{emo}"] = None

                results.append(row)

            frame_idx += 1
            pbar.update(1)

        cap.release()
        pbar.close()

        df = pd.DataFrame(results)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        if output_csv_path is None:
            #base_name = os.path.splitext(os.path.basename(video_path))[0]
            # al guardar:
            #base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_csv = f"{base_name}_emotions.csv"      
        else:
            output_csv = os.path.join(output_csv_path, f"{base_name}_emotions.csv")
            
        df.to_csv(output_csv, index=False)
        print(f"Saved results to {output_csv}")
        return df


def smooth_data(df, window_size=5):
    """
    Rolling average over prob_* columns (centered).
    """
    if df is None or df.empty:
        return df

    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    df_smoothed = df.copy()
    df_smoothed[prob_cols] = (
        df[prob_cols].rolling(window=window_size, min_periods=1, center=True).mean()
    )
    return df_smoothed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepFace Video Emotion Analyzer")
    parser.add_argument("--videos", nargs="+", required=True, help="List of video paths")
    parser.add_argument("--step", type=int, default=5, help="Process every Nth frame (default: 5)")
    parser.add_argument("--smooth", type=int, default=0, help="Window size for smoothing (0 disables)")
    # en argparse:
    parser.add_argument("--outdir", type=str, default=".", help="Output directory")

    parser.add_argument(
        "--backend",
        type=str,
        default="opencv",
        help="Face detection backend (opencv, retinaface, mtcnn, etc.)",
    )
    parser.add_argument(
        "--enforce-detection",
        action="store_true",
        help="If set, fail frames when no face is detected (default: False).",
    )

    args = parser.parse_args()

    # ✅ FIX: backend (and enforce_detection) now actually applied
    analyzer = VideoEmotionAnalyzer(
        detector_backend=args.backend,
        enforce_detection=args.enforce_detection,
    )

    for video_file in args.videos:
        df = analyzer.process_video(video_file,process_every_n_frames=args.step,
        output_csv_path= args.outdir)

        if args.smooth > 0 and df is not None:
            print(f"Applying smoothing (window={args.smooth})...")
            df_smooth = smooth_data(df, window_size=args.smooth)
            base_name = os.path.splitext(os.path.basename(video_file))[0]
            smooth_out = os.path.join(args.outdir, f"{base_name}_emotions_smoothed.csv")
            df_smooth.to_csv(smooth_out, index=False)
            print(f"Saved smoothed results to {smooth_out}")

