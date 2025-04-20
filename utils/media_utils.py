import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import mimetypes
from io import BytesIO


def detect_media_type(uploaded_file):
    """
    Detect whether the uploaded file is an image or video.
    Returns: "image", "video", or "unknown"
    """
    # First: check file extension using MIME
    mime_type, _ = mimetypes.guess_type(uploaded_file.name)

    if mime_type:
        if mime_type.startswith("image"):
            return "image"
        elif mime_type.startswith("video"):
            return "video"

    # Fallback: try to open as image
    try:
        image = Image.open(BytesIO(uploaded_file.getvalue()))
        image.verify()
        return "image"
    except UnidentifiedImageError:
        return "video"  # assume video if PIL fails
    except Exception:
        return "unknown"

def extract_video_thumbnail(file_bytes):
    """
    Extracts the first frame of a video and returns:
    - PIL thumbnail image
    - Encoded JPEG bytes for analysis
    """
    import tempfile

    nparr = np.frombuffer(file_bytes, np.uint8)
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    with open(temp_path, "wb") as f:
        f.write(nparr)

    cap = cv2.VideoCapture(temp_path)
    success, frame = cap.read()
    cap.release()

    if not success:
        return None, None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    thumbnail = Image.fromarray(frame_rgb)

    # Encode to bytes for Azure analysis
    _, encoded = cv2.imencode(".jpg", frame_rgb)
    return thumbnail, encoded.tobytes()
