import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
from pathlib import Path
from dlvc.datasets.cifar10 import CIFAR10Dataset
from dlvc.metrics import Accuracy
from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
# for wandb users:
from dlvc.wandb_logger import WandBLogger
#import os

class BaseTrainer(metaclass=ABCMeta):
    '''
    Base class of all Trainers.
    '''

    @abstractmethod
    def train(self) -> None:
        '''
        Holds training logic.
        '''

        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float, float]:
        '''
        Holds validation logic for one epoch.
        '''

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float, float]:
        '''
        Holds training logic for one epoch.
        '''

        pass


class ImgClassificationTrainer(BaseTrainer):
    def __init__(self, 
                 model, 
                 optimizer,
                 loss_fn,
                 lr_scheduler,
                 train_metric,
                 val_metric,
                 train_data,
                 val_data,
                 device,
                 num_epochs: int, 
                 training_save_dir: Path,
                 batch_size: int = 4,
                 val_frequency: int = 5) -> None:
        
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.num_epochs = num_epochs
        self.training_save_dir = training_save_dir
        self.batch_size = batch_size
        self.val_frequency = val_frequency
        self.train_accuracy_history = []
        self.val_accuracy_history = []

        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        self.best_val_mPCAcc = 0.0

    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        #correct_predictions = 0


        for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader, desc=f"Training Epoch {epoch_idx}", leave=False)):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            
            # Convert targets to torch.long
            targets = targets.long()

            #forward pass
            outputs = self.model(inputs)

            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)


            self.train_metric.update(outputs, targets)

        epoch_loss = total_loss / total_samples
        print("epoch_loss",epoch_loss)

        epoch_mAcc,epoch_mPCAcc, epoch_mPCAcc_3   = self.train_metric.accuracy(), self.train_metric.per_class_accuracy(), self.train_metric.per_class_accuracies() 
        print("epoch_accuracy , epoch_per_class_accuracy", epoch_mAcc, epoch_mPCAcc)

        #for i, acc in enumerate(epoch_mPCAcc_3):
        #    print(f"Accuracy for class {self.val_data.classes[i]}: {acc:.4f}")

        return epoch_loss, epoch_mAcc, epoch_mPCAcc

    def _val_epoch(self, epoch_idx:int) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        #correct_predictions = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(self.val_loader, desc=f"Validation Epoch {epoch_idx}", leave=False)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)

                targets = targets.to(torch.long)


                loss = self.loss_fn(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

                # Update accuracy metric
                self.val_metric.update(outputs, targets)

        epoch_loss = total_loss / total_samples
        epoch_mAcc, epoch_mPCAcc, epoch_mPCAcc_3 = self.val_metric.accuracy(), self.val_metric.per_class_accuracy(), self.val_metric.per_class_accuracies()
        print("validation_epoch_works ---------------------------")
        print("epoch_validation accuracy and per class", epoch_mAcc, epoch_mPCAcc)
        print("-----------------------------")
        for i, acc in enumerate(epoch_mPCAcc_3):
            print(f"Accuracy for class {self.val_data.classes[i]}: {acc:.4f}")
        print("-----------------------------")

        return epoch_loss, epoch_mAcc ,epoch_mPCAcc #epoch_accuracy,

    def train(self) -> None:
        for epoch_idx in range(self.num_epochs):
            train_loss, train_accuracy, train_mPCAcc = self._train_epoch(epoch_idx)
            val_loss, val_accuracy, val_mPCAcc = self._val_epoch(epoch_idx)

            ###############
            # Append accuracy values to history lists
            self.train_accuracy_history.append(train_accuracy)
            self.val_accuracy_history.append(val_accuracy)
            ############

            print(f"________________________epoch    {epoch_idx}________________________ ")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(f"Train per class Acc: {train_mPCAcc:.4f}")  # Here, use train_mPCAcc
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            print(f"Val per class Acc: {val_mPCAcc:.4f}")  # Here, use val_mPCAcc


            if val_mPCAcc > self.best_val_mPCAcc:
                self.best_val_mPCAcc = val_mPCAcc
                save_path = self.training_save_dir / "best_model.pth"
                torch.save(self.model.state_dict(), save_path)
                print(save_path)
                print(f"Model saved at: {save_path}")
            print("________________________epoch ends________________________ ")    

    def save_accuracy_history(self):
        # Save accuracy history lists to a file or any other storage
        # You can use libraries like numpy or pickle to save the lists
        np.savez('accuracy_history.npz', train_accuracy=self.train_accuracy_history, val_accuracy=self.val_accuracy_history)
        #pass
    def plot_accuracy_history(self, train_accuracy_history, val_accuracy_history):
    
    #Plot the accuracy histor
        epochs = range(1, len(train_accuracy_history) + 1)
        plt.plot(epochs, train_accuracy_history, 'r', label='Training Accuracy')
        plt.plot(epochs, val_accuracy_history, 'b', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        #plt.show()
        plt.savefig('accuracy_plot.png')
        plt.show()