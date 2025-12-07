import streamlit as st
from PIL import Image
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from streamlit_image_coordinates import streamlit_image_coordinates
from io import BytesIO

st.title("Click → Segment → Cut Object using Meta SAM (No OpenCV)")

# Load SAM model (CPU or GPU)
@st.cache_resource
def load_sam():
    sam_checkpoint = "sam_vit_h_4b8939.pth"  # Replace with your SAM checkpoint
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)
    return predictor

predictor = load_sam()

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
if not uploaded_file:
    st.info("Upload an image to begin")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
img_np = np.array(image)

st.write("Click on the image to select an object point")
coords = streamlit_image_coordinates(img_np)

if coords:
    cx, cy = coords["x"], coords["y"]
    st.success(f"Clicked at: ({cx}, {cy})")

    # Prepare SAM predictor
    predictor.set_image(img_np)
    input_point = np.array([[cx, cy]])
    input_label = np.array([1])  # positive point

    # Predict mask
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    mask = masks[np.argmax(scores)]  # Best mask

    # Apply mask to cut object
    cut = img_np.copy()
    cut[~mask] = 0

    st.write("### Segmented Object")
    st.image(cut, use_column_width=True)

    # Download as PNG
    cut_pil = Image.fromarray(cut)
    buf = BytesIO()
    cut_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        "Download cut object",
        data=byte_im,
        file_name="cut_object.png",
        mime="image/png"
    )
else:
    st.write("Click somewhere on the image to segment an object.")
