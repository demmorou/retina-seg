import torch, cv2, albumentations as A
import segmentation_models_pytorch as smp

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"PyTorch {torch.__version__} | OpenCV {cv2.__version__} | device: {device}")

model = smp.Unet("efficientnet-b3", encoder_weights="imagenet", in_channels=3, classes=4).to(device)
x = torch.randn(1, 3, 512, 512, device=device)
with torch.no_grad():
    y = model(x)
print("Saída:", tuple(y.shape))  # esperado: (1, 4, 512, 512)