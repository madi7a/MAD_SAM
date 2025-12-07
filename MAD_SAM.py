import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import SAM
from streamlit_image_coordinates import streamlit_image_coordinates

st.title("Click → Segment → Cut Object using Ultralytics‑SAM")

# Load SAM model
@st.cache_resource
def load_sam(model_path="sam_b.pt"):
    sam = SAM(model_path)
    return sam

sam_model = load_sam("sam_b.pt")  # ensure you have the pretrained .pt file in working dir

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
    # Note: SAM expects list-of-points, labels
    results = sam_model.predict(
        img_np, 
        points=[ [cx, cy] ],
        labels=[1],
        multimask_output=True
    )

    # The results contain masks; we pick the first (or highest confidence) mask
    if len(results.masks) > 0:
        mask = results.masks[0]  # numpy array of bool / 0/1 mask
        # Apply mask to image
        cut = img_np.copy()
        # If mask is boolean or 0/1, convert to boolean index
        cut[~mask.astype(bool)] = 0

        st.write("### Segmented Object")
        st.image(cut, use_column_width=True)

        # Optionally let user download
        cut_pil = Image.fromarray(cut)
        buf = cv2.imencode('.png', cut[:, :, ::-1])[1].tobytes()
        st.download_button("Download cut object", data=buf, file_name="object.png", mime="image/png")
    else:
        st.error("No mask was returned by SAM.")
else:
    st.write("Click somewhere on the image to segment an object.")
