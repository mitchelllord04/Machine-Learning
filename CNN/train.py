import torch
import torch.nn as nn

from model import Model
from data import get_data

def main():
    # Pick the best available device
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )
    print("Using device:", device)


    EPOCHS = 100

    train_loader, test_loader = get_data()

    model = Model().to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Put model in training mode
    model.train()

    # Best accuracy recorded across all epochs
    best = 0.0

    # Write header for stats file
    with open("stats.txt", "w") as f:
        f.write("epoch,mean_loss,training_accuracy,validation_accuracy\n")

    # Training loop
    for i in range(EPOCHS):
        running_loss = 0
        correct_train = 0
        total_train = 0

        # Iterate over all training batches
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Delete any old gradients
            optimizer.zero_grad()

            # Make predictions
            logits = model(images)

            # Compute loss
            loss = loss_function(logits, labels)
            running_loss += loss.item() * labels.size(0)


            predictions = logits.argmax(dim=1)
            correct_train += (predictions == labels).sum().item()
            total_train += labels.size(0)

            # Back-propagate
            loss.backward()
            optimizer.step()

        avg_train_loss = running_loss / total_train
        train_accuracy = correct_train / total_train

        model.eval()

        correct = 0
        total = 0

        # Compute test accuracy
        # Disable gradient tracking for faster eval
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                predictions = logits.argmax(dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        test_accuracy = correct / total
        if test_accuracy > best:
            best = test_accuracy
            torch.save(model.state_dict(), "model.pth")

        print(
            f"Epoch {i + 1}: Train Loss: {avg_train_loss:.4f}, "
            f"Train Acc: {train_accuracy:.4f}, "
            f"Test Acc: {test_accuracy:.4f}, Best: {best:.4f}"
        )

        # Record stats for plotting loss and accuracy
        with open("stats.txt", "a") as f:
            f.write(f"{i+1},{avg_train_loss},{train_accuracy},{test_accuracy}\n")

        model.train()
        scheduler.step()

if __name__ == '__main__':
    main()