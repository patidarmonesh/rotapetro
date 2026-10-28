"""MOCK — not part of V1, not used for any result"""

import cv2
import numpy as np
import imageio
import torch
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

class VideoInference:
    """Phase 11 (§17): Full inference pipeline for PPL/XPL video inputs."""
    
    def __init__(self, model_path: str = None, cfg: Dict[str, Any] = None, device: str = 'cpu', **kwargs):
        if cfg is None and "cfg" in kwargs:
            cfg = kwargs["cfg"]
        if cfg is not None and (cfg.get("v1") is not None or cfg.get("v1_mode")):
            raise NotImplementedError("infer_video is a mock and not implemented for V1")
        self.cfg = cfg
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        
    def _load_model(self, path: str):
        # Stub logic to load architecture, weights omitted for brevity but API is standard
        # return torch.load(path, map_location=self.device)
        return None

    def process(self, ppl_video: str, xpl_video: str, meta: Dict[str, Any], output_dir: str):
        """Full pipeline: decode, align, segment, infer, report."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        ppl_frames = self.decode_video(ppl_video)
        xpl_frames = self.decode_video(xpl_video)
        
        qc_info = self.quality_checks(xpl_frames)
        
        center, radius = self.detect_field(ppl_frames[0])
        angles = self.estimate_angles(ppl_frames)
        selected_indices = self.pick_angles(angles, ppl_frames)
        
        # Inference logic goes here (segmentation, classification)
        # using the selected frames matching [0, 5, ..., 175] degrees
        
        # Mock grains payload for report
        grains = [{"id": 1, "class": "Quartz", "confidence": 0.95}]
        
        report = self.generate_report(grains, meta.get('field_id', 'unknown'), qc_info)
        with open(out_path / f"{meta.get('field_id', 'unknown')}_report.json", 'w') as f:
            json.dump(report, f, indent=2)

    def decode_video(self, video_path: str) -> List[np.ndarray]:
        """Decode using imageio/cv2."""
        reader = imageio.get_reader(video_path)
        frames = []
        for i, frame in enumerate(reader):
            # frame is RGB
            frames.append(frame)
        reader.close()
        return frames

    def quality_checks(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Blur (Laplacian var), clipping (<5% at 255), exposure drift (<5%)."""
        blur_scores = [cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for f in frames]
        is_blurry = np.mean(blur_scores) < 100.0  # Threshold example
        
        # Clipping: max value pixels
        clipping_pcts = [(np.sum(f >= 250) / f.size) * 100 for f in frames]
        has_clipping = any(p > 5.0 for p in clipping_pcts)
        
        # Exposure drift
        mean_intensities = [np.mean(f) for f in frames]
        drift = (max(mean_intensities) - min(mean_intensities)) / max(min(mean_intensities), 1) * 100
        has_drift = drift > 5.0
        
        return {
            "blur_mean": float(np.mean(blur_scores)),
            "is_blurry": bool(is_blurry),
            "max_clipping_pct": float(max(clipping_pcts)),
            "has_clipping": bool(has_clipping),
            "exposure_drift_pct": float(drift),
            "has_exposure_drift": bool(has_drift)
        }

    def detect_field(self, frame: np.ndarray) -> Tuple[Tuple[int, int], int]:
        """Hough circle for dark border -> center, radius."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
            param1=50, param2=30, minRadius=int(frame.shape[0]*0.3), maxRadius=int(frame.shape[0]*0.6)
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            c = circles[0, 0]
            return (int(c[0]), int(c[1])), int(c[2])
            
        return (frame.shape[1]//2, frame.shape[0]//2), frame.shape[0]//2

    def estimate_angles(self, frames: List[np.ndarray]) -> List[float]:
        """ORB+RANSAC to first frame, chain for >60°, unwrap."""
        orb = cv2.ORB_create()
        angles = [0.0]
        
        gray0 = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        kp0, des0 = orb.detectAndCompute(gray0, None)
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        ref_idx = 0
        current_angle = 0.0
        
        for i in range(1, len(frames)):
            gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            kp, des = orb.detectAndCompute(gray, None)
            
            matches = bf.match(des0, des)
            if len(matches) < 10:
                angles.append(current_angle)
                continue
                
            src_pts = np.float32([kp0[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
            if M is not None:
                # Extract rotation from affine matrix
                angle_rad = np.arctan2(M[1, 0], M[0, 0])
                angle_deg = np.degrees(angle_rad)
                # Unwrap and accumulate
                diff = (angle_deg - current_angle + 180) % 360 - 180
                current_angle += diff
                
            angles.append(current_angle)
            
            # Re-anchor if drift is high (>60 degrees from ref)
            if abs(current_angle) > 60:
                gray0 = gray
                kp0, des0 = kp, des
                ref_idx = i
                
        return [abs(a) % 360 for a in angles]

    def pick_angles(self, angles: List[float], frames: List[np.ndarray]) -> List[int]:
        """Nearest to 0,5,...,175° with <=3° gap."""
        target_angles = np.arange(0, 180, 5)
        selected_indices = []
        
        angles_arr = np.array(angles)
        
        for target in target_angles:
            diffs = np.abs(angles_arr - target)
            min_idx = int(np.argmin(diffs))
            if diffs[min_idx] <= 3.0:
                selected_indices.append(min_idx)
            else:
                # Fallback, just pick closest even if > 3 deg
                selected_indices.append(min_idx)
                
        return selected_indices

    def generate_report(self, grains: List[Dict[str, Any]], field_id: str, qc_info: Dict[str, Any]) -> Dict[str, Any]:
        """JSON per §17.3 format."""
        return {
            "schema_version": "1.0",
            "metadata": {
                "field_id": field_id,
                "qc": qc_info
            },
            "minerals_detected": list(set(g['class'] for g in grains)),
            "grains": grains
        }


def main(cfg: Dict[str, Any] = None):
    """Entry point for video inference."""
    if cfg is not None and (cfg.get("v1") is not None or cfg.get("v1_mode")):
        raise NotImplementedError("infer_video is a mock and not implemented for V1")
    raise NotImplementedError("infer_video is a mock and not implemented for V1")


if __name__ == "__main__":
    main()

