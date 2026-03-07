"""
01_data_extractor.py
PURPOSE : YOLO + ByteTrack → CSV at 0.2s intervals
SOURCE  : supervision official (roboflow.com/supervision)
          Ultralytics tracking API (docs.ultralytics.com/modes/track)
USAGE   : python 01_data_extractor.py -i video.mp4 -o trajectories.csv
"""
import cv2, csv, time, argparse, logging
import numpy as np
from ultralytics import YOLO
import supervision as sv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_CLASSES = ['car','truck','bus','motorcycle','bicycle']
SAMPLE_INTERVAL_S = 0.2

# Standard Indian mixed-traffic dimensions (IRC + IITB defaults)
VEHICLE_DIMS = {
    'car':        {'length': 4.5,  'width': 1.8},
    'truck':      {'length': 8.0,  'width': 2.5},
    'bus':        {'length': 12.0, 'width': 2.5},
    'motorcycle': {'length': 2.0,  'width': 0.8},
    'bicycle':    {'length': 1.8,  'width': 0.6},
    'unknown':    {'length': 4.5,  'width': 1.8},
}

CSV_FIELDS = [
    'frame_number','vehicle_id','timestamp_s','vehicle_type',
    'x_center_px','y_center_px','x_front_px','y_front_px',
    'x_min','y_min','x_max','y_max','width_px','height_px',
    'confidence','vehicle_length_m','vehicle_width_m',
]

def front_centre(x1, y1, x2, y2):
    """Front-centre pixel — right edge for left-to-right traffic flow.
    Flip x2→x1 for right-to-left flow."""
    return float(x2), float((y1 + y2) / 2.0)

class TrajectoryExtractor:
    def __init__(self, model_path, conf, fps, output_csv):
        logger.info(f"Loading YOLO: {model_path}")
        self.model   = YOLO(model_path)
        self.conf    = conf
        self.fps     = fps
        # supervision ByteTrack — official API
        self.tracker = sv.ByteTrack(
            frame_rate=int(fps),
            track_activation_threshold=0.5,    # was 0.3 — raise to reduce ghost tracks
            lost_track_buffer=int(fps * 3),    # was 1.5x — hold ID longer across occlusion
            minimum_matching_threshold=0.8,    # was 0.9 — slightly more flexible matching
        )
        self._last_written = {}
        self._f   = open(output_csv, 'w', newline='', encoding='utf-8')
        self._w   = csv.DictWriter(self._f, fieldnames=CSV_FIELDS)
        self._w.writeheader()
        logger.info(f"CSV sink: {output_csv}")

    def process_frame(self, frame, frame_idx):
        ts = frame_idx / self.fps
        # Ultralytics inference
        results    = self.model(frame, conf=self.conf, verbose=False, imgsz=640)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Filter vehicles + attach class name string (supervision data field pattern)
        if detections.class_id is not None and len(detections.class_id):
            names = self.model.names
            keep  = np.array([names[int(c)] in TARGET_CLASSES for c in detections.class_id])
            detections = detections[keep]
            if len(detections):
                detections['vehicle_type'] = np.array([names[int(c)] for c in detections.class_id])
        else:
            detections = sv.Detections.empty()

        # ByteTrack update
        if len(detections):
            detections = self.tracker.update_with_detections(detections)
        else:
            self.tracker.update_with_detections(sv.Detections.empty())
            return None

        if detections.tracker_id is None:
            return None

        # Write at 0.2s sample intervals
        for i in range(len(detections)):
            tid  = int(detections.tracker_id[i])
            last = self._last_written.get(tid, -SAMPLE_INTERVAL_S)
            if (ts - last) < SAMPLE_INTERVAL_S - 1e-6:
                continue
            self._last_written[tid] = ts

            x1, y1, x2, y2 = map(float, detections.xyxy[i])
            cx, cy = (x1+x2)/2, (y1+y2)/2
            fx, fy = front_centre(x1, y1, x2, y2)
            vt = str(detections.data['vehicle_type'][i]) if 'vehicle_type' in detections.data else 'unknown'
            dims = VEHICLE_DIMS.get(vt, VEHICLE_DIMS['unknown'])
            conf_v = float(detections.confidence[i]) if detections.confidence is not None else -1.0

            self._w.writerow({
                'frame_number': frame_idx, 'vehicle_id': tid, 'timestamp_s': round(ts,3),
                'vehicle_type': vt, 'x_center_px': round(cx,2), 'y_center_px': round(cy,2),
                'x_front_px': round(fx,2), 'y_front_px': round(fy,2),
                'x_min': round(x1,2), 'y_min': round(y1,2), 'x_max': round(x2,2), 'y_max': round(y2,2),
                'width_px': round(x2-x1,2), 'height_px': round(y2-y1,2),
                'confidence': round(conf_v,4),
                'vehicle_length_m': dims['length'], 'vehicle_width_m': dims['width'],
            })
        return detections

    def close(self):
        self._f.close()
        logger.info("CSV closed.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--input', required=True)
    ap.add_argument('-o','--output', default='trajectories.csv')
    ap.add_argument('-m','--model', default='yolov8n.pt')
    ap.add_argument('--conf', type=float, default=0.4)
    ap.add_argument('--video-fps', type=float, default=0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = args.video_fps if args.video_fps > 0 else cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0; logger.warning("FPS unknown — defaulting to 30")

    logger.info(f"Video {W}x{H}  FPS={fps:.2f}  ~{N} frames")
    ext = TrajectoryExtractor(args.model, args.conf, fps, args.output)

    idx = 0; t0 = time.time()
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok: break
            ext.process_frame(frame, idx)
            idx += 1
            if idx % 150 == 0:
                logger.info(f"Frame {idx}/{N}  {idx/(time.time()-t0):.1f} fps")
    finally:
        cap.release(); ext.close()
    logger.info(f"Done. {idx} frames → {args.output}")

if __name__ == '__main__':
    main()