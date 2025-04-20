# backend/text_processor.py

import cv2
import numpy as np
import time
from io import BytesIO
from PIL import Image
from difflib import get_close_matches
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
AZURE_ENDPOINT = "https://4382-cv.cognitiveservices.azure.com/"
AZURE_KEY = "DjCee3Pxd3UOWxvUvc1pe4JA0pGOf0PVqEOMTE8f5eDMVyrzRGmEJQQJ99BDACGhslBXJ3w3AAAFACOGfrpH"

# Azure client
computervision_client = ComputerVisionClient(
    AZURE_ENDPOINT, CognitiveServicesCredentials(AZURE_KEY)
)


def preprocess_image(image_bytes):
    img_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, encoded_img = cv2.imencode(".jpg", threshed)
    return encoded_img.tobytes()


def extract_text(image_bytes):
    preprocessed_bytes = preprocess_image(image_bytes)

    try:
        poller = computervision_client.read_in_stream(image=BytesIO(preprocessed_bytes), raw=True)
        operation_id = poller.headers["Operation-Location"].split("/")[-1]
    except Exception as e:
        print("Azure OCR request failed:", e)
        return []

    for _ in range(10):
        result = computervision_client.get_read_result(operation_id)
        if result.status.lower() in ['succeeded', 'failed']:
            break
        time.sleep(1)

    lines = []
    if result.status.lower() == 'succeeded':
        for page in result.analyze_result.read_results:
            for line in page.lines:
                lines.append(line.text)

    return lines


def correct_ocr_text(lines, vocabulary, threshold=0.8):
    corrected = []
    for word in lines:
        match = get_close_matches(word, vocabulary, n=1, cutoff=threshold)
        corrected.append(match[0] if match else word)
    return corrected


def visualize_ocr_on_image(image_bytes):
    preprocessed_bytes = preprocess_image(image_bytes)

    try:
        poller = computervision_client.read_in_stream(image=BytesIO(preprocessed_bytes), raw=True)
        operation_id = poller.headers["Operation-Location"].split("/")[-1]
    except Exception as e:
        print("Azure OCR request failed:", e)
        return None

    for _ in range(10):
        result = computervision_client.get_read_result(operation_id)
        if result.status.lower() in ['succeeded', 'failed']:
            break
        time.sleep(1)

    np_img = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if result.status.lower() == 'succeeded':
        for page in result.analyze_result.read_results:
            for line in page.lines:
                points = line.bounding_box
                pts = np.array([[int(points[i]), int(points[i + 1])] for i in range(0, len(points), 2)], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                x, y = pts[0][0]
                cv2.putText(img, line.text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    print("OCR completed but returned no readable content.")
    return None
