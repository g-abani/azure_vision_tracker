# app.py
import streamlit as st
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import tempfile
import streamlit.components.v1 as components

from backend.image_processor import process_image
from backend.video_processor import process_video
from backend.text_processor import extract_text, visualize_ocr_on_image
from utils.media_utils import detect_media_type, extract_video_thumbnail

st.title("Azure Vision AI - Image & Video Analyzer")

uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "mov"])

if uploaded_file:
    media_type = detect_media_type(uploaded_file)
    st.write(f"🧠 Detected media type: `{media_type}`")

    if media_type == "image":
        analysis_mode = st.radio("Select Analysis Mode:", ["Objects", "Text (OCR)"])
        image_bytes = uploaded_file.getvalue()

        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if analysis_mode == "Objects":
                with st.spinner("Analyzing image for objects..."):
                    result_image, objects, _ = process_image(uploaded_file)
                st.success("Object detection complete ✅")
                st.image(result_image, caption="Detected Objects", use_container_width=True)
                st.write("Detected Objects:", objects)

            elif analysis_mode == "Text (OCR)":
                with st.spinner("Extracting text and drawing boxes..."):
                    annotated_image = visualize_ocr_on_image(image_bytes)
                    text_lines = extract_text(image_bytes)

                if annotated_image:
                    st.success("OCR complete ✅")
                    st.image(annotated_image, caption="OCR Result with Bounding Boxes", use_container_width=True)
                else:
                    st.error("Failed to visualize OCR results — possibly an API error or no text found.")

                if text_lines:
                    st.subheader("📄 Extracted Text")
                    for line in text_lines:
                        st.markdown(f"- {line}")

        except UnidentifiedImageError:
            st.error("Uploaded file is not a valid image format. Please upload .jpg or .png.")

    elif media_type == "video":
        video_bytes = uploaded_file.getvalue()
        thumbnail, _ = extract_video_thumbnail(video_bytes)
        if thumbnail:
            st.image(thumbnail, caption="🎬 Video Thumbnail", use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            video_path = tmp.name

        with st.spinner("Analyzing video... This may take a few seconds..."):
            annotated_path, log_path, viz_path, gif_path, webm_path = process_video(video_path)
        st.success("Video analysis complete ✅")
        print(annotated_path)
        #with open(annotated_path, "rb") as video_file:
        #    st.video(video_file.read())

        st.subheader("⚡ Preview Formats")
        col1, col2 = st.columns(2)

        with col1:
            st.image(gif_path, caption="GIF Preview", use_container_width=True)
            with open(gif_path, "rb") as f:
                st.download_button("⬇️ Download GIF", f, file_name="annotated_preview.gif")

        with col2:
            with open(webm_path, "rb") as f:
                st.video(f.read(), format="video/webm")
                st.download_button("⬇️ Download WebM", f, file_name="annotated_preview.webm")

        if viz_path:
            st.subheader("📊 Object Movement Visualization")
            with open(viz_path, "r", encoding="utf-8") as f:
                html = f.read()
            components.html(html, height=600, scrolling=True)

        st.download_button("Download Position Log", open(log_path, "rb"), file_name="position_log.csv")

    else:
        st.error("Unsupported or unrecognized file type. Please upload a valid image or video file.")

    if st.button("🔄 Clear Results"):
        st.rerun()
