from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import torch
import numpy as np
import base64
import uvicorn
import cv2
from PIL import Image
from io import BytesIO


from model import DenseNetChestXRayModel, NIH14_CLASSES
from image_utils import preprocess_image_for_inference

# --- Utility for Visualization ---
def show_cam_on_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """ 
    Overlays a heatmap onto a numpy image.
    This applies a colormap to the raw heatmap and merges it with the original image.
    """
    # Ensure mask is properly shaped and normalized
    mask_normalized = np.clip(mask, 0, 1)
    mask_uint8 = (255 * mask_normalized).astype(np.uint8)
    heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    result = (255 * cam).astype(np.uint8)  # type: ignore
    return result  # type: ignore


MODEL_WEIGHTS_PATH = "NIHDenseNet.pth"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using device: {device.type.upper()} ---")


# --- FastAPI App Setup ---
app = FastAPI(title="Rad-Insight API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading ---
# Instantiate the model
model = DenseNetChestXRayModel(num_classes=len(NIH14_CLASSES), pretrained=True)

try:
    # **UPDATED**: Load the weights directly onto the selected device
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    print(f"--- Successfully loaded custom model weights from '{MODEL_WEIGHTS_PATH}' ---")
except FileNotFoundError:
    print(f"\n--- WARNING: Custom weights '{MODEL_WEIGHTS_PATH}' not found. Using ImageNet base. ---")
except Exception as e:
    print(f"An unexpected error occurred loading model weights: {e}")


model.to(device)
model.eval()
print(f"--- Rad-Insight Model is loaded on {device.type.upper()} and ready for inference. ---")

# --- API Response Models ---
class AnalysisResponse(BaseModel):
    probabilities: Dict[str, float]
    heatmap_image: Optional[str]

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "Rad-Insight API is running. Ready to analyze."}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_cxr(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    image_bytes = await image.read()
    
    try:
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        input_tensor = preprocess_image_for_inference(image_bytes)
        
        # **NEW**: Move the input tensor to the same device as the model (GPU)
        input_tensor = input_tensor.to(device)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {e}")

    try:
        # No changes needed here; tensor and model are on the same device
        probs = model.predict_proba(input_tensor)
        # Squeeze and move to CPU for NumPy conversion
        probs_list = probs.cpu().numpy().squeeze() 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference error: {str(e)}")

    # Get all probabilities with their class names
    all_probabilities = {label: float(prob) for label, prob in zip(NIH14_CLASSES, probs_list)}
    
    # Sort probabilities in descending order and get top 5
    sorted_probs = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)
    top_5_probabilities = dict(sorted_probs[:5])

    heatmap_b64_string = None
    if np.any(probs_list):
        highest_prob_index = int(np.argmax(probs_list))
        
        print(f"Generating heatmap for top prediction: '{NIH14_CLASSES[highest_prob_index]}'")
        
        heatmap_tensor, _ = model.grad_cam(input_tensor, class_index=highest_prob_index)
        
        
        heatmap_np = heatmap_tensor.squeeze().cpu().numpy()
        original_image_np = np.array(pil_image.resize((224, 224))) / 255.0
        
        overlay = show_cam_on_image(original_image_np, heatmap_np)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode(".jpg", overlay_bgr)
        
        if success:
            heatmap_b64_string = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

    return AnalysisResponse(probabilities=top_5_probabilities, heatmap_image=heatmap_b64_string)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)