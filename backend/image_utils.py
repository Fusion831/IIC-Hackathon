

import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO

# Normalization constants defined in the PRD (Step T2 & I2)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TARGET_SIZE = (224, 224)


train_transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.RandomRotation(15),          # <Transform name="RandomRotation" degrees="15" />
    transforms.RandomHorizontalFlip(p=0.5), # <Transform name="RandomHorizontalFlip" probability="0.5" />
    transforms.ColorJitter(brightness=0.1, contrast=0.1), # <Transform name="ColorJitter" ... />
    transforms.ToTensor(),                  # Converts image to [0, 1] range tensor
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) 
])


inference_transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def preprocess_image_for_inference(image_bytes: bytes) -> torch.Tensor:
    """
    Takes raw image bytes (from an API upload), processes them using
    the INFERENCE transform, and returns a batch-ready tensor.
    """
    try:
        image = Image.open(BytesIO(image_bytes))
        
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        
        image_resized = transforms.Resize(TARGET_SIZE)(image)
        tensor = transforms.ToTensor()(image_resized)
        tensor = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(tensor)
        
        # Add a batch dimension (models expect [BatchSize, C, H, W])
        return tensor.unsqueeze(0)  # Shape: [1, 3, 224, 224]
        
    except Exception as e:
        # Properly handle the exception by raising it with more context
        raise RuntimeError(f"Error preprocessing image: {e}") from e