# utils/azure_api.py

import cv2
import os
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from difflib import get_close_matches
from io import BytesIO

# Load environment variables
# load_dotenv()

AZURE_ENDPOINT = "https://4382-computer-vision.cognitiveservices.azure.com/"
AZURE_KEY = "CkVzy8Fy4OhsyBt6SAEUl8DuAE8kSZqL6PQtOM6D6VPL6hZigttFJQQJ99BDACYeBjFXJ3w3AAAFACOGXp9x"

if not AZURE_ENDPOINT or not AZURE_KEY:
    raise ValueError("Missing AZURE_ENDPOINT or AZURE_KEY.")

# Authenticate SDK client
computervision_client = ComputerVisionClient(
    AZURE_ENDPOINT,
    CognitiveServicesCredentials(AZURE_KEY)
)

def analyze_image(image_bytes):
    """
    Analyzes an image using Azure Computer Vision Client SDK to detect objects.
    
    Returns:
        List of dicts with object name and bounding rectangle.
    """
    image_stream = BytesIO(image_bytes)
    analysis = computervision_client.analyze_image_in_stream(
        image=image_stream,
        visual_features=[VisualFeatureTypes.objects]
    )

    result = []
    for obj in analysis.objects:
        result.append({
            "object": obj.object_property,
            "confidence": obj.confidence,
            "rectangle": {
                "x": int(obj.rectangle.x),
                "y": int(obj.rectangle.y),
                "w": int(obj.rectangle.w),
                "h": int(obj.rectangle.h)
            }
        })

    return result
# utils/azure_api.py

def extract_text(image_bytes):
    """
    Performs OCR using Azure Read API v3.2 and returns detected text lines.
    """
    image_stream = BytesIO(image_bytes)

    # Step 1: Submit image for OCR
    poller = computervision_client.read_in_stream(image=image_stream, raw=True)

    # Step 2: Extract Operation-Location to get operation ID
    operation_location = poller.headers["Operation-Location"]
    operation_id = operation_location.split("/")[-1]

    # Step 3: Poll for result using operation ID
    result = computervision_client.get_read_result(operation_id)

    while result.status not in ['succeeded', 'failed']:
        result = computervision_client.get_read_result(operation_id)

    # Step 4: Extract text lines if succeeded
    lines = []
    if result.status == 'succeeded':
        for page in result.analyze_result.read_results:
            for line in page.lines:
                lines.append(line.text)

    return lines


from difflib import get_close_matches

def correct_ocr_text(lines, vocabulary, threshold=0.8):
    corrected = []
    for word in lines:
        match = get_close_matches(word, vocabulary, n=1, cutoff=threshold)
        corrected.append(match[0] if match else word)
    return corrected

def analyze_tags(image_bytes):
    image_stream = BytesIO(image_bytes)
    analysis = computervision_client.analyze_image_in_stream(
        image=image_stream,
        visual_features=["Tags"]
    )
    tags = []
    for tag in analysis.tags:
        tags.append(f"{tag.name} ({tag.confidence * 100:.1f}%)")
    return tags
