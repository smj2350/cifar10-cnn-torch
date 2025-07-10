import torch
import matplotlib.pyplot as plt
from CNN_train import SimpleCNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def visualize_predictions(model, test_loader, num_samples=5):
    model.eval()
    x_sample = []
    t_sample = []
    predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            x_sample.extend(images.cpu())
            t_sample.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())

            if len(x_sample) >= num_samples:
                break

    for i in range(num_samples):
        plt.figure(figsize=(2, 2))
        plt.imshow(x_sample[i].squeeze(), cmap='gray')
        plt.title(f"Prediction: {predictions[i]} (True: {t_sample[i]})")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("cnn_model.pth"))

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    visualize_predictions(model, test_loader, num_samples=10)