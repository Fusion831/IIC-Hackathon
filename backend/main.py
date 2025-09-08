

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
import random 
import uvicorn


try:
    from .image_utils import preprocess_image_for_inference
except ImportError:
    from image_utils import preprocess_image_for_inference


# This MUST be in the same order the final model is trained on
PATHOLOGY_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]

app = FastAPI(title="Rad-Insight API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


model = None 



@app.get("/")
def read_root():
    return {"status": "Rad-Insight API is running. Ready to analyze."}


@app.post("/analyze")
async def analyze_cxr(image: UploadFile = File(...)):
    
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    
    image_bytes = await image.read()
    tensor = preprocess_image_for_inference(image_bytes)
    
    if tensor is None:
       raise HTTPException(status_code=500, detail="Error processing the image.")

    
    if model:
        
        try:
            with torch.no_grad(): 
                logits = model(tensor)             
                probs = torch.sigmoid(logits)      
                probs_list = probs.cpu().numpy()[0] 
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model inference error: {str(e)}")
    else:
        
        print("--- WARNING: Model not loaded. Generating DUMMY data. ---")
        probs_list = [random.uniform(0.01, 0.99) for _ in PATHOLOGY_LABELS]
        
        probs_list[1] = 0.95 # Cardiomegaly
        probs_list[4] = 0.91 # Effusion
        probs_list[13] = 0.02 # Pneumothorax
        

    
    # Zip the labels and probabilities together into a dict
    results = {label: prob for label, prob in zip(PATHOLOGY_LABELS, probs_list)}
    
    print(f"Returning results: {results}")
    return results

if __name__ == "__main__":
    uvicorn.run("0.0.0.0",port = 8000)

