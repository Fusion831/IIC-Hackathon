import os
from dotenv import load_dotenv
import uvicorn
import base64
import numpy as np
import torch
import cv2
from PIL import Image
from io import BytesIO
from collections import OrderedDict

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("--- WARNING: httpx not installed. AI report generation will be disabled. ---")

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List


from model import DenseNetChestXRayModel, primary_model, backup_model
from image_utils import preprocess_image_for_inference


load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


if GEMINI_API_KEY:
    
    masked_key = GEMINI_API_KEY[:8] + "..." + GEMINI_API_KEY[-4:] if len(GEMINI_API_KEY) > 12 else "***"
    print(f"--- GEMINI API key loaded successfully (masked: {masked_key}) ---")
else:
    print("--- WARNING: GEMINI_API_KEY not found in environment variables ---")
    print("--- Make sure your .env file contains: GEMINI_API_KEY=your_actual_key ---")


def show_cam_on_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """ Overlays a heatmap onto a numpy image using a weighted blend. """
    
    mask_normalized = np.clip(mask, 0, 1)
    mask_uint8 = (255 * mask_normalized).astype(np.uint8)
    
    
    heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    
    img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    
    
    blended_image = cv2.addWeighted(img_uint8, 0.6, heatmap, 0.4, 0)
    return blended_image


async def generate_radiology_report(probabilities: Dict[str, float]) -> Optional[str]:
    """ Calls the Gemini API to generate a structured radiology report from model probabilities. """
    if not HTTPX_AVAILABLE:
        print("--- ERROR: httpx not available for API calls ---")
        return "AI report generation is disabled. Please install httpx: pip install httpx"
    
    if not GEMINI_API_KEY:
        print("--- ERROR: GEMINI_API_KEY not configured ---")
        return "AI report generation is disabled. Please configure the API key via an environment variable."

    print("--- Calling Gemini API for report generation ---")
    gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    
    sorted_probs = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    top_finding = sorted_probs[0]
    other_notable = [f"{p[0]} ({p[1]:.1%})" for p in sorted_probs[1:4] if p[1] > 0.2]

    prompt = f"""
    You are an AI assistant helping a radiologist draft a report based on findings from a computer vision model.
    Your task is to generate a structured, professional summary. Do NOT invent clinical details.
    
    The AI model's top predicted finding is: {top_finding[0]} with a confidence of {top_finding[1]:.1%}.
    Other notable findings with moderate confidence include: {', '.join(other_notable) if other_notable else 'None'}.

    Based ONLY on this information, generate a report with the following sections:
    - FINDINGS: A one-paragraph summary of the AI model's predictions.
    - IMPRESSION: A one-sentence conclusion based on the primary finding.
    - RECOMMENDATION: A brief, generic recommendation for clinical correlation.
    """

    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            print(f"--- Making API request to Gemini ---")
            response = await client.post(gemini_api_url, json=payload)
            
            print(f"--- Gemini API response status: {response.status_code} ---")
            
            if response.status_code != 200:
                print(f"--- ERROR: Gemini API returned status {response.status_code} ---")
                print(f"--- Response: {response.text} ---")
                return f"Error: Gemini API returned status {response.status_code}"
            
            response.raise_for_status()
            result = response.json()
            
            # Safely navigate the JSON response
            report_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", None)
            
            if report_text:
                print("--- Successfully generated AI report ---")
                return report_text
            else:
                print("--- ERROR: No text content in Gemini response ---")
                return "Error: No content returned from Gemini API"
                
    except httpx.TimeoutException:
        print("--- ERROR: Gemini API request timed out ---")
        return "Error: API request timed out"
    except httpx.RequestError as e:
        print(f"--- ERROR: Network error calling Gemini API: {e} ---")
        return "Error: Network error occurred"
    except Exception as e:
        print(f"--- ERROR calling Gemini API: {e} ---")
        return "Error: Could not generate AI report."


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using device: {device.type.upper()} ---")

# Initialize models
current_model = None
model_type = "none"

# Try to load primary fine-tuned model first
if primary_model.model is not None:
    current_model = primary_model
    model_type = "fine_tuned"
    print(f"--- Primary fine-tuned DenseNet loaded successfully ---")
else:
    # Try to load backup model if primary failed
    print("--- Primary model failed, attempting to load backup model ---")
    backup_loaded = backup_model.load_model("NIHDenseNet.pth")
    if backup_loaded:
        current_model = backup_model
        model_type = "backup"
        print(f"--- Backup original DenseNet loaded successfully ---")
    else:
        print("--- ERROR: No models could be loaded! ---")
        # Fallback to basic model initialization
        current_model = DenseNetChestXRayModel()
        current_model.to(device)
        current_model.eval()
        model_type = "basic"
        print(f"--- Fallback to basic DenseNet with ImageNet weights ---")

print(f"--- Active model type: {model_type.upper()} ---")
print(f"--- Model is ready for inference on {device.type.upper()} ---")


app = FastAPI(title="Rad-Insight API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class AnalysisResponse(BaseModel):
    probabilities: Dict[str, float]
    heatmap_image: Optional[str]
    report_text: Optional[str]

@app.get("/")
def read_root():
    return {"status": "Rad-Insight API is running with Live Inference and NLP Reporting."}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_cxr(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    image_bytes = await image.read()
    
    try:
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        input_tensor = preprocess_image_for_inference(image_bytes).to(device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {e}")

    
    try:
        # Check if current_model is available
        if current_model is None:
            raise HTTPException(status_code=500, detail="No model available for inference")
            
        # Use current_model which is either primary_model or backup_model
        if hasattr(current_model, 'predict_proba'):
            # For wrapper models (FineTunedDenseNet or OriginalDenseNet)
            probs = current_model.predict_proba(input_tensor)
        else:
            # For direct DenseNetChestXRayModel instances
            probs = current_model.predict_proba(input_tensor)
            
        probs_list = probs.cpu().numpy().squeeze()
        
        # Get class names from the current model
        if hasattr(current_model, 'class_names'):
            class_names = current_model.class_names
        else:
            from model import NIH14_CLASSES
            class_names = NIH14_CLASSES
            
        all_probabilities = {label: float(prob) for label, prob in zip(class_names, probs_list)}
        
        sorted_probs = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)
        top_5_probabilities = dict(sorted_probs[:5])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference error: {str(e)}")
    
    
    report_text = await generate_radiology_report(all_probabilities)

    
    heatmap_b64_string = None
    try:
        if current_model is not None:
            highest_prob_index = int(np.argmax(probs_list))
            
            # Generate heatmap using current model
            if hasattr(current_model, 'grad_cam'):
                # For wrapper models (FineTunedDenseNet or OriginalDenseNet)
                heatmap_tensor, _ = current_model.grad_cam(input_tensor, class_index=highest_prob_index)
            else:
                # For direct DenseNetChestXRayModel instances
                heatmap_tensor, _ = current_model.grad_cam(input_tensor, class_index=highest_prob_index)
            
            heatmap_np = heatmap_tensor.squeeze().cpu().numpy()
            original_image_np = np.array(pil_image.resize((224, 224))) / 255.0
            
            overlay = show_cam_on_image(original_image_np, heatmap_np)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            success, buffer = cv2.imencode(".jpg", overlay_bgr)
            
            if success:
                heatmap_b64_string = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    except Exception as e:
        print(f"--- ERROR generating Grad-CAM: {e} ---")

    return AnalysisResponse(
        probabilities=top_5_probabilities,
        heatmap_image=heatmap_b64_string,
        report_text=report_text
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


