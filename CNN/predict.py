import sys
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import Model

# Take in an image, run it through the model, and predict its class

CLASSES = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Initialize model
model = Model().to(device)
state = torch.load("model.pth", map_location=device)
model.model.load_state_dict(state)
model.eval()

# Resize and normalize image
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)

# Get image path from command line argument
image_path = sys.argv[1]
image = Image.open(image_path).convert("RGB")

x = transform(image).unsqueeze(0).to(device)

# Make prediction
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    pred = probs.argmax().item()

print(f'Prediction: {CLASSES[pred]}')
print(f"Confidence: {probs[pred].item() :.4f}")
