from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import pickle
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


NIH14_CLASSES: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]


class DenseNetChestXRayModel(nn.Module):
    """
    DenseNet-based multi-label classifier for NIH ChestX-ray14.
    - Multi-label probabilities via sigmoid
    - Grad-CAM heatmaps for visual explanation
    """

    def __init__(
        self,
        num_classes: int = len(NIH14_CLASSES),
        model_variant: str = "densenet121",
        pretrained: bool = True,
        in_channels: int = 3,
        class_names: Optional[List[str]] = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names or NIH14_CLASSES[:num_classes]
        self.model_variant = model_variant
        self.in_channels = in_channels

        # Select weights handle per variant
        weights = None
        if pretrained:
            if model_variant == "densenet121":
                weights = models.DenseNet121_Weights.IMAGENET1K_V1
            elif model_variant == "densenet169":
                weights = models.DenseNet169_Weights.IMAGENET1K_V1
            elif model_variant == "densenet201":
                weights = models.DenseNet201_Weights.IMAGENET1K_V1

        # Build backbone with graceful fallback if weight download fails
        def build_backbone(selected_weights):
            if model_variant == "densenet121":
                return models.densenet121(weights=selected_weights)
            elif model_variant == "densenet169":
                return models.densenet169(weights=selected_weights)
            elif model_variant == "densenet201":
                return models.densenet201(weights=selected_weights)
            else:
                raise ValueError(f"Unsupported DenseNet variant: {model_variant}")

        try:
            backbone = build_backbone(weights)
        except Exception:
            
            backbone = build_backbone(None)

        feature_size = backbone.classifier.in_features

        if in_channels != 3:
            first_conv = backbone.features.conv0
            if not isinstance(first_conv, nn.Conv2d):
                raise TypeError("Expected first layer to be Conv2d")
                
            # Extract attributes with proper type handling
            kernel_size = first_conv.kernel_size
            stride = first_conv.stride  
            padding = first_conv.padding
            
            new_conv = nn.Conv2d(
                in_channels,
                first_conv.out_channels,
                kernel_size=kernel_size,  # type: ignore
                stride=stride,  # type: ignore
                padding=padding,  # type: ignore
                bias=first_conv.bias is not None,
            )
            with torch.no_grad():
                if in_channels > 3 and first_conv.weight.shape[1] == 3:
                    repeat_factor = (in_channels + 2) // 3
                    repeated = first_conv.weight.repeat(1, repeat_factor, 1, 1)[:, :in_channels]
                    new_conv.weight.copy_(repeated / (in_channels / 3.0))
                elif in_channels == 1 and first_conv.weight.shape[1] == 3:
                    gray_weights = first_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight.copy_(gray_weights)
                else:
                    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
                if new_conv.bias is not None:
                    nn.init.zeros_(new_conv.bias)
            backbone.features.conv0 = new_conv

        self.backbone: models.DenseNet = backbone
        self.backbone.classifier = nn.Linear(feature_size, num_classes)

        if freeze_backbone:
            for p in self.backbone.features.parameters():
                p.requires_grad = False

        self._grad_cam_target_layer_name = "features.norm5"
        self._cached_activations: Optional[torch.Tensor] = None
        self._cached_gradients: Optional[torch.Tensor] = None
        self._register_grad_cam_hooks()

        self.eval_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def _find_module_by_name(self, module: nn.Module, name: str) -> nn.Module:
        current: nn.Module = module
        for attr in name.split("."):
            current = getattr(current, attr)
        return current

    def _register_grad_cam_hooks(self) -> None:
        target_layer = self._find_module_by_name(self.backbone, self._grad_cam_target_layer_name)

        def forward_hook(_m: nn.Module, _inp: Tuple[torch.Tensor], out: torch.Tensor) -> None:
            # Cache activations
            self._cached_activations = out.detach()
            
            if out.requires_grad:
                def save_grad(grad: torch.Tensor) -> None:
                    self._cached_gradients = grad.detach()
                out.register_hook(save_grad)

       
        target_layer.register_forward_hook(forward_hook)

    def grad_cam(
        self,
        images: torch.Tensor,
        class_index: Optional[int] = None,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (heatmaps[B,1,H,W], selected_class_probs[B])
        """
        self.eval()
        images = images.requires_grad_(True)
        logits = self.forward(images)
        probs = torch.sigmoid(logits)

        if class_index is None:
            target_scores, _ = probs.max(dim=1)
        else:
            target_scores = probs[:, class_index]

        self.zero_grad(set_to_none=True)
        target_scores.sum().backward(retain_graph=True)

        assert self._cached_activations is not None, "Grad-CAM activations not captured"
        assert self._cached_gradients is not None, "Grad-CAM gradients not captured"

        activations = self._cached_activations
        gradients = self._cached_gradients

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)

        if normalize:
            cam_min = cam.amin(dim=(2, 3), keepdim=True)
            cam_max = cam.amax(dim=(2, 3), keepdim=True)
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam, target_scores.detach()

    @torch.inference_mode()
    def infer_topk(self, images: torch.Tensor, k: int = 5) -> List[Dict[str, float]]:
        probs = self.predict_proba(images)
        results: List[Dict[str, float]] = []
        for prob in probs:
            values, indices = torch.topk(prob, k=min(k, self.num_classes))
            results.append({self.class_names[int(i)]: float(v) for v, i in zip(values, indices)})
        return results


def create_default_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class FineTunedDenseNet:
    """
    Primary fine-tuned DenseNet model wrapper that maintains all functionality
    including heatmap generation and grad-cam capabilities
    """
    
    def __init__(self, model_path: str = "densenet_chestxray.pkl"):
        self.model: Optional[DenseNetChestXRayModel] = None
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = NIH14_CLASSES
        self.load_model()
    
    def load_model(self) -> bool:
        """Load the fine-tuned DenseNet model from PKL"""
        try:
            if not os.path.exists(self.model_path):
                print(f"Fine-tuned model not found at {self.model_path}")
                return False
                
            # Try loading with torch.load first (for PyTorch models)
            try:
                loaded_data = torch.load(self.model_path, map_location='cpu')
            except Exception:
                # Fallback to pickle.load if torch.load fails
                with open(self.model_path, 'rb') as f:
                    loaded_data = pickle.load(f)
            
            # Check if it's already a DenseNetChestXRayModel instance
            if isinstance(loaded_data, DenseNetChestXRayModel):
                self.model = loaded_data
            elif isinstance(loaded_data, dict):
                # Check if it has nested state_dict (common in saved models)
                if 'state_dict' in loaded_data:
                    # Extract the actual state_dict
                    state_dict = loaded_data['state_dict']
                    
                    # Extract other metadata if available
                    num_classes = loaded_data.get('num_classes', len(self.class_names))
                    variant = loaded_data.get('variant', 'densenet121')
                    
                    # Create model and load state dict
                    self.model = DenseNetChestXRayModel(
                        num_classes=num_classes,
                        model_variant=variant,
                        pretrained=False,
                        class_names=self.class_names
                    )
                    self.model.load_state_dict(state_dict)
                else:
                    # If it's a direct state dict, create model and load it
                    self.model = DenseNetChestXRayModel(
                        num_classes=len(self.class_names),
                        model_variant="densenet121",
                        pretrained=False,
                        class_names=self.class_names
                    )
                    self.model.load_state_dict(loaded_data)
            elif hasattr(loaded_data, 'state_dict'):
                # If it's a PyTorch model with state_dict method
                self.model = DenseNetChestXRayModel(
                    num_classes=len(self.class_names),
                    model_variant="densenet121",
                    pretrained=False,
                    class_names=self.class_names
                )
                self.model.load_state_dict(loaded_data.state_dict())
            else:
                print(f"Unsupported model format in PKL file: {type(loaded_data)}")
                return False
            
            # Move to appropriate device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"Fine-tuned DenseNet loaded successfully from {self.model_path}")
            print(f"Using device: {self.device}")
            return True
            
        except Exception as e:
            print(f"Error loading fine-tuned DenseNet: {e}")
            return False
    
    def predict_proba(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Make prediction using the fine-tuned DenseNet model"""
        if self.model is None:
            raise ValueError("Fine-tuned model not loaded")
        
        # Move tensor to same device as model
        image_tensor = image_tensor.to(self.device)
        
        # Use the model's predict_proba method
        return self.model.predict_proba(image_tensor)
    
    def grad_cam(self, image_tensor: torch.Tensor, class_index: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate grad-cam heatmap using the fine-tuned model"""
        if self.model is None:
            raise ValueError("Fine-tuned model not loaded")
        
        # Move tensor to same device as model
        image_tensor = image_tensor.to(self.device)
        
        # Use the model's grad_cam method
        return self.model.grad_cam(image_tensor, class_index=class_index)
    
    def get_model(self) -> Optional[DenseNetChestXRayModel]:
        """Return the underlying model"""
        return self.model


class OriginalDenseNet:
    """
    Backup original DenseNet model with same interface as FineTunedDenseNet
    """
    
    def __init__(self):
        self.model: Optional[DenseNetChestXRayModel] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = NIH14_CLASSES
    
    def load_model(self, model_path: str = "NIHDenseNet.pth") -> bool:
        """Load original PyTorch model as backup"""
        try:
            # Create model instance
            self.model = DenseNetChestXRayModel(
                num_classes=len(self.class_names),
                model_variant="densenet121",
                pretrained=False,
                class_names=self.class_names
            )
            
            # Load state dict
            if torch.cuda.is_available():
                state_dict = torch.load(model_path)
            else:
                state_dict = torch.load(model_path, map_location='cpu')
            
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print("Backup original DenseNet loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading backup model: {e}")
            return False
    
    def predict_proba(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Backup prediction method"""
        if self.model is None:
            raise ValueError("Backup model not loaded")
        
        # Move tensor to same device as model
        image_tensor = image_tensor.to(self.device)
        
        # Use the model's predict_proba method
        return self.model.predict_proba(image_tensor)
    
    def grad_cam(self, image_tensor: torch.Tensor, class_index: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate grad-cam heatmap using the original model"""
        if self.model is None:
            raise ValueError("Backup model not loaded")
        
        # Move tensor to same device as model
        image_tensor = image_tensor.to(self.device)
        
        # Use the model's grad_cam method
        return self.model.grad_cam(image_tensor, class_index=class_index)
    
    def get_model(self) -> Optional[DenseNetChestXRayModel]:
        """Return the underlying model"""
        return self.model


# Initialize model instances
primary_model = FineTunedDenseNet()
backup_model = OriginalDenseNet()