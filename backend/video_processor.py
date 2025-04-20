
import cv2
import os
import csv
import time
import numpy as np
import tempfile
import plotly.graph_objects as go
import subprocess
from utils.azure_api import analyze_image
from backend.tracker import CentroidTracker

def process_video(video_path, frame_interval=10, delay_between_requests=1.0):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = temp_video.name
    temp_log = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    log_path = temp_log.name
    temp_html = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    viz_path = temp_html.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    tracker = CentroidTracker(max_distance=50)
    trails = {}
    colors = {}

    def get_color(object_id):
        np.random.seed(object_id)
        return tuple(int(c) for c in np.random.randint(100, 255, 3))

    frame_id = 0

    with open(log_path, mode='w', newline='') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["frame", "object_id", "label", "x", "y", "w", "h"])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            if frame_id % frame_interval != 0:
                continue

            _, img_encoded = cv2.imencode('.jpg', frame)
            try:
                objects = analyze_image(img_encoded.tobytes())
            except Exception as e:
                print(f"Azure error at frame {frame_id}: {e}")
                continue

            detections = [{
                "label": obj["object"],
                "x": obj["rectangle"]["x"],
                "y": obj["rectangle"]["y"],
                "w": obj["rectangle"]["w"],
                "h": obj["rectangle"]["h"]
            } for obj in objects]

            tracked = tracker.update(detections)

            for object_id, info in tracked.items():
                label = info["label"]
                x, y, w, h = info["bbox"]
                cx, cy = info["centroid"]
                tag = f"{label}_{object_id}"

                trails.setdefault(tag, []).append((cx, cy))
                colors.setdefault(tag, get_color(object_id))

                color = colors[tag]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, tag, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                log_writer.writerow([frame_id, object_id, label, x, y, w, h])

                pts = trails[tag][-10:]
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i - 1], pts[i], color, 1)

            out.write(frame)
            time.sleep(delay_between_requests)

    cap.release()
    out.release()

    # Plotly chart
    fig = go.Figure()
    for tag, pts in trails.items():
        if pts:
            xs, ys = zip(*pts)
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode='lines+markers',
                name=tag,
                marker=dict(size=6)
            ))

    fig.update_layout(
        title="Object Trajectories",
        xaxis_title="X Position",
        yaxis_title="Y Position",
        template="plotly_white"
    )
    fig.write_html(viz_path)

    # Export GIF and WebM
    gif_path = out_path.replace(".mp4", ".gif")
    webm_path = out_path.replace(".mp4", ".webm")

    subprocess.run([
        "ffmpeg", "-y", "-i", out_path,
        "-vf", "fps=10,scale=640:-1:flags=lanczos",
        "-loop", "0", gif_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    subprocess.run([
        "ffmpeg", "-y", "-i", out_path,
        "-c:v", "libvpx-vp9", "-b:v", "1M",
        "-c:a", "libopus", webm_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return out_path, log_path, viz_path, gif_path, webm_path
