import csv
import matplotlib.pyplot as plt

epochs = []
loss = []
train_acc = []
val_acc = []

with open("stats.txt", "r") as f:
    reader = csv.reader(f)

    # skip header
    next(reader)

    for row in reader:
        if len(row) < 4:
            continue

        epochs.append(int(row[0]))
        loss.append(float(row[1]))
        train_acc.append(float(row[2]))
        val_acc.append(float(row[3]))

fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=150)

# Loss plot
axs[0].plot(epochs, loss)
axs[0].set_title("Training Loss")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Mean Cross-Entropy Loss")

# Accuracy plot
axs[1].plot(epochs, train_acc, label="Training")
axs[1].plot(epochs, val_acc, label="Validation")
axs[1].set_ylim(0.5, 1.0)
axs[1].grid(alpha=0.3)
axs[1].set_title("Accuracy")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Accuracy")
axs[1].legend()

plt.tight_layout()

fig.savefig("learning.png")
plt.show()
