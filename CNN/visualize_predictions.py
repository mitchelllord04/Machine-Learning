import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt

from model import Model

# Compare predictions with actual images from the dataset

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model().to(device)
state = torch.load("model.pth", map_location=device)
model.model.load_state_dict(state)
model.eval()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

dataset = CIFAR10(root="./data", train=False, download=False, transform=None)

fig, axes = plt.subplots(4, 4, figsize=(8, 8))

for ax in axes.flat:

    idx = torch.randint(0, len(dataset), (1,)).item()
    image, label = dataset[idx]

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred = probs.argmax().item()

    ax.imshow(image)
    ax.set_title(f"{CLASSES[pred]} ({probs[pred]:.2f})")
    ax.axis("off")

plt.tight_layout()
fig.savefig("visualized.png")
plt.show()
