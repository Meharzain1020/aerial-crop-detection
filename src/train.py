import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

def train_model(model, dataloader, epochs=7, device="cpu", output_weights="outputs/weights/fine_tuned_crop_detector.pth"):
    os.makedirs(os.path.dirname(output_weights), exist_ok=True)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0003)

    loss_history = []

    for ep in range(epochs):
        model.train()
        total_loss = 0.0

        for images, targets, _ in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        avg_loss = total_loss / max(1, len(dataloader))
        loss_history.append(avg_loss)
        print(f"Epoch [{ep+1}/{epochs}] - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), output_weights)

    os.makedirs("outputs/figures", exist_ok=True)
    fig = plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), loss_history, marker='o', color='forestgreen', linewidth=2)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("outputs/figures/training_loss.png")
    plt.close()

    return model