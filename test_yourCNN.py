import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from torchvision.models import resnet18
from dlvc.models.class_model import DeepClassifier
from dlvc.metrics import Accuracy
from dlvc.datasets.cifar10 import CIFAR10Dataset
from torch.utils.data import DataLoader
from dlvc.models.cnn import YourCNN

def test(args):

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset
    test_data = CIFAR10Dataset('dlvc/datasets/cifar-10-python.tar.gz', 'test', transform=transform)
    test_data_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_test_data = len(test_data)

    # Load trained model
    model = DeepClassifier(YourCNN(num_classes=test_data.num_classes()))
    
    model.load_state_dict(torch.load(args.path_to_trained_model))
    model.to(device)
    
    # Define loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Initialize accuracy metric
    test_metric = Accuracy(classes=test_data.classes)
    
    # Testing loop
    test_loss = 0.0
    total_samples = 0
    correct_predictions = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets.long())
            test_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            correct_predictions += torch.sum(torch.argmax(outputs, dim=1) == targets).item()
            test_metric.update(outputs, targets)
    
    # Compute final metrics
    test_loss /= total_samples
    test_accuracy = correct_predictions / total_samples
    test_mAcc, test_mPCAcc, test_mPCAcc_3 = test_metric.accuracy(), test_metric.per_class_accuracy(), test_metric.per_class_accuracies()
    
    # Print final results
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_mAcc:.4f}")
    print(f"Per class accuracy:2 {test_mPCAcc:.4f}")
    for i, acc in enumerate(test_mPCAcc_3):
        print(f"Accuracy for class {test_data.classes[i]}: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for testing')
    parser.add_argument('--path_to_trained_model', type=str, default="saved_models/best_model.pth", help='Path to the trained model')
    args = parser.parse_args()

    test(args)

