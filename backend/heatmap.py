import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
from io import BytesIO
from typing import Optional, Any

def generate_heatmap_overlay(model: Any, target_layer: Any, input_tensor: torch.Tensor, 
                           original_image_bytes: bytes, target_class_index: int) -> Optional[bytes]:
    """
    Generates a Grad-CAM heatmap and overlays it on the original image.
    
    This function creates a class activation map using Grad-CAM to visualize which parts
    of the image the model focuses on when making predictions. The process involves:
    
    1. Creating a GradCAM object with the model and target layer
    2. Computing gradients with respect to the target class
    3. Generating a grayscale heatmap showing important regions
    4. Decoding the original image from bytes
    5. Resizing and normalizing the original image
    6. Overlaying the heatmap on the original image
    7. Converting back to BGR and encoding as JPEG bytes
    
    Args:
        model: The PyTorch model (e.g., our RadInsightModel or a test ImageNet model).
        target_layer: The specific convolutional layer to target (e.g., model.base_model.features[-1]).
        input_tensor: The fully preprocessed, batch-ready tensor (shape [1, 3, 224, 224]).
        original_image_bytes: The raw bytes of the *original* uploaded image (for high-res overlay).
        target_class_index: The integer index of the class we want to explain.
        
    Returns:
        Bytes of the final overlayed JPG image, or None if it fails.
    """
    targets = [ClassifierOutputTarget(target_class_index)]

    try:
        with GradCAM(model=model, target_layers=[target_layer]) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)  # type: ignore
            
            grayscale_cam = grayscale_cam[0, :]

            image_array = np.frombuffer(original_image_bytes, np.uint8)
            original_image_rgb = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if original_image_rgb is None:
                print("Error: Could not decode the original image")
                return None
            
            original_image_rgb = cv2.cvtColor(original_image_rgb, cv2.COLOR_BGR2RGB)

            original_image_resized = cv2.resize(original_image_rgb, (224, 224))
            
            rgb_img_float = np.array(original_image_resized, dtype=np.float32) / 255.0

            overlay = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            (success, encoded_image) = cv2.imencode(".jpg", overlay_bgr)
            
            if success:
                return encoded_image.tobytes()
            else:
                print("Error: Failed to encode the overlay image")
                return None

    except Exception as e:
        print(f"Error during Grad-CAM generation: {e}")
        return None