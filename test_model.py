import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

weights = MobileNet_V2_Weights.DEFAULT

model = mobilenet_v2(weights=weights)
model = model.to(device)

print("Device:", device)
print("Model: MobileNetV2")
print("Pretrained weights loaded successfully!")

total_params = sum(p.numel() for p in model.parameters())

print(f"Total parameters: {total_params:,}")