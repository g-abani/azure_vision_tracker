# backend/video_processor.py

import cv2
import os
import csv
import time
import numpy as np
import tempfile
import plotly.graph_objects as go
from utils.azure_api import analyze_image


def process_video(video_path, frame_interval=10, delay_between_requests=1.0):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Temporary output video file
    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = temp_video_file.name

    # Temporary log and chart file
    temp_log_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    log_path = temp_log_file.name

    temp_html_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    viz_path = temp_html_file.name

    # OpenCV video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1', 'H264'
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    positions = {}  # {label: [(x, y)]}
    colors = {}     # consistent color per label

    def get_color(label):
        import random
        if label not in colors:
            colors[label] = tuple(random.randint(100, 255) for _ in range(3))
        return colors[label]

    frame_count = 0

    with open(log_path, mode='w', newline='') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["frame", "object", "x", "y", "w", "h"])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            _, img_encoded = cv2.imencode('.jpg', frame)
            try:
                objects = analyze_image(img_encoded.tobytes())
            except Exception as e:
                print(f"⚠️ Azure API error at frame {frame_count}: {e}")
                continue

            for obj in objects:
                label = obj["object"]
                bbox = obj["rectangle"]
                x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
                cx, cy = x + w // 2, y + h // 2

                positions.setdefault(label, []).append((cx, cy))

                color = get_color(label)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                log_writer.writerow([frame_count, label, x, y, w, h])

            out.write(frame)
            time.sleep(delay_between_requests)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Plotly interactive movement chart
    fig = go.Figure()
    for label, coords in positions.items():
        if coords:
            xs, ys = zip(*coords)
            frame_nums = [i * frame_interval for i in range(len(xs))]
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode='lines+markers+text',
                name=label,
                text=[f"{label} - Frame {f}" for f in frame_nums],
                hoverinfo='text+x+y',
                line=dict(width=2),
                marker=dict(size=6)
            ))

    fig.update_layout(
        title="📍 Object Position Changes Across Frames",
        xaxis_title="X Position (pixels)",
        yaxis_title="Y Position (pixels)",
        legend_title="Detected Objects",
        template="plotly_white",
        height=600
    )
    fig.write_html(viz_path)

    return out_path, log_path, viz_path