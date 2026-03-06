import random
from torchvision.datasets import CIFAR10

# Get a random image from the CIFAR10 dataset

dataset = CIFAR10(root="./data", train=False, download=False, transform=None)

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

index = random.randrange(len(dataset))

image, label = dataset[index]

filename = f"sample_{index}_{classes[label]}.png"
image.save(filename)

print("Saved:", filename)
print("Label:", classes[label])
