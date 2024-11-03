import streamlit as st
import cv2
import numpy as np
from model_inference import display

st.title("Object Detection Web App")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Getting the image name from the uploaded file
        image_name = uploaded_image.name
        
        # Calling the display function with both image and image name
        processed_image, precision, recall = display(image, image_name)

        # Displaying the processed image
        st.image(processed_image, caption="Processed Image", use_column_width=True)

        # Displaying metrics
        if precision > 0 or recall > 0:
            st.write(f"Precision: {precision:.2f}")
            st.write(f"Recall: {recall:.2f}")
        else:
            st.info("No metrics available for this image. This might be due to no detections or missing ground truth data.")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.error("Please make sure the image name matches the ground truth data in the CSV file.")