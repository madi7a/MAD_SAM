import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import SAM
from streamlit_image_coordinates import streamlit_image_coordinates
from io import BytesIO

st.title("Click → Segment → Cut Object using Ultralytics‑SAM (No OpenCV)")

# Load SAM model
@st.cache_resource
def load_sam(model_path="sam_b.pt"):
    return SAM(model_path)

sam_model = load_sam("sam_b.pt")  # ensure you have the pretrained .pt file

# Upload image
uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
if not uploaded:
    st.info("Upload an image to begin")
    st.stop()

img = Image.open(uploaded).convert("RGB")
img_np = np.array(img)

st.write("Click on the image to select an object point")
coords = streamlit_image_coordinates(img_np)

if coords:
    cx, cy = coords["x"], coords["y"]
    st.success(f"Clicked at: ({cx}, {cy})")

    # Run SAM segmentation with point prompt
    results = sam_model.predict(
        img_np, 
        points=[[cx, cy]],
        labels=[1],
        multimask_output=True
    )

    if len(results.masks) > 0:
        mask = results.masks[0].astype(bool)

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
            file_name="object.png",
            mime="image/png"
        )
    else:
        st.error("No mask returned by SAM.")
