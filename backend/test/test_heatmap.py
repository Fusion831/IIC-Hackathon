import torch
import torchvision.models as models
from PIL import Image
from io import BytesIO
import os

# --- HACKATHON NOTE ---
# We are assuming the following files are in the same directory or in the python path:
# - image_utils.py 
# - heatmap.py (renamed from your original snippet for clarity)
# --------------------
try:
    from image_utils import preprocess_image_for_inference
    # Let's assume your heatmap function is in a file named `heatmap.py`
    from heatmap import generate_heatmap_overlay
except ImportError:
    print("ERROR: Make sure 'image_utils.py' and 'heatmap.py' are in the same directory.")
    exit()

# The local test image provided by the user
TEST_IMAGE_FILENAME = "test_image.png"

def run_heatmap_test_local():
    """
    Runs the Grad-CAM test using a local image file and a pre-trained
    ImageNet model to validate the heatmap generation logic.
    """
    print("--- STARTING HEATMAP TEST (USING LOCAL IMAGE & ImageNet MODEL) ---")

    # 1. CHECK FOR IMAGE
    if not os.path.exists(TEST_IMAGE_FILENAME):
        print(f"\nFATAL ERROR: Test image not found!")
        print(f"Please make sure the image file named '{TEST_IMAGE_FILENAME}' is in the same folder as this script.")
        return

    # 2. LOAD MODEL
    print("Loading pre-trained ImageNet DenseNet-121...")
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.eval()

    # 3. DEFINE TARGETS
    target_layer = model.features[-1]
    IMAGENET_TARGET_CLASS_INDEX = 805  # Class index for "stethoscope"
    print(f"Model loaded. Target layer: final features. Target class: {IMAGENET_TARGET_CLASS_INDEX} (stethoscope)")

    # 4. LOAD IMAGE BYTES
    with open(TEST_IMAGE_FILENAME, "rb") as f:
        image_bytes = f.read()
    print(f"Loaded '{TEST_IMAGE_FILENAME}' from disk.")

    # 5. PREPROCESS IMAGE
    input_tensor = preprocess_image_for_inference(image_bytes)
    if input_tensor is None:
        print("ERROR: Preprocessing failed.")
        return
    print(f"Image preprocessed into tensor shape: {input_tensor.shape}")

    # 6. GENERATE HEATMAP
    print("Generating Grad-CAM overlay...")
    overlay_bytes = generate_heatmap_overlay(
        model=model,
        target_layer=target_layer,
        input_tensor=input_tensor,
        original_image_bytes=image_bytes,
        target_class_index=IMAGENET_TARGET_CLASS_INDEX
    )

    # 7. SAVE OUTPUT
    if overlay_bytes:
        output_filename = "heatmap_output_local.jpg"
        with open(output_filename, "wb") as f:
            f.write(overlay_bytes)
        print(f"\nSUCCESS! Saved heatmap to '{output_filename}'.")
        print("Check the file to see the visual overlay. It should highlight the stethoscope.")
    else:
        print("\nFAILURE: Heatmap generation returned None.")

if __name__ == "__main__":
    run_heatmap_test_local()

