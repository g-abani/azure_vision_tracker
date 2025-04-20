# backend/image_processor.py

import cv2
import numpy as np
from utils.azure_api import analyze_image
from PIL import Image, UnidentifiedImageError
from utils.azure_api import analyze_image, extract_text, analyze_tags
import io

def process_image(uploaded_file):
    try:
        img_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(image)
        tags = analyze_tags(img_bytes)
        objects = analyze_image(img_bytes)

        for obj in objects:
            bbox = obj["rectangle"]
            label = obj["object"]
            confidence = obj.get("confidence", 0) * 100
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

            # Draw bounding box and label with confidence
            cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label_text = f"{label} ({confidence:.1f}%)"
            cv2.putText(img_np, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)

        # Show all detected objects with confidence
        object_labels = [f"{obj['object']} ({obj['confidence']*100:.1f}%)" for obj in objects]
        detected_text = extract_text(img_bytes)
        # return Image.fromarray(img_np), object_labels
        return Image.fromarray(img_np), [f"{obj['object']} ({obj['confidence']*100:.1f}%)" for obj in objects], detected_text

    except UnidentifiedImageError:
        raise ValueError("Uploaded file is not a valid image. Please upload a .jpg, .png, or .jpeg file.")
