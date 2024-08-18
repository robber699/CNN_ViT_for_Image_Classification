from abc import ABCMeta, abstractmethod
import torch
from typing import List


class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass



class Accuracy(PerformanceMeasure):
    '''
    Average classification accuracy.
    '''

    def __init__(self, classes) -> None:
        self.classes = classes
        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        self.total_samples = 0
        self.correct_predictions = 0
        self.class_correct = [0] * len(self.classes)
        self.class_total = [0] * len(self.classes)

    def update(self, prediction: torch.Tensor, 
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (s,c) with each row being a class-score vector.
        target must have shape (s,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        '''

        if prediction.shape[0] != target.shape[0]:
            raise ValueError("Prediction and target shapes do not match")

        for i in range(len(target)):
            pred_class = torch.argmax(prediction[i])
            true_class = int(target[i])

            self.total_samples += 1
            self.class_total[true_class] += 1

            if pred_class == true_class:
                self.correct_predictions += 1
                self.class_correct[true_class] += 1

    def __str__(self):
        '''
        Return a string representation of the performance, accuracy and per class accuracy.
        '''

        accuracy = self.accuracy()
        per_class_accuracy = self.per_class_accuracy()

        return f"Accuracy: {accuracy:.4f}\nPer class accuracy: {per_class_accuracy:.4f}"

    def accuracy(self) -> float:
        '''
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''
        if self.total_samples == 0:
            return 0.0
        return self.correct_predictions / self.total_samples

    def per_class_accuracy(self) -> float:
        '''
        Compute and return the per class accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''
        class_accuracy = [0.0] * len(self.classes)
        for i in range(len(self.classes)):
            if self.class_total[i] != 0:
                class_accuracy[i] = self.class_correct[i] / self.class_total[i]
        return sum(class_accuracy) / len(self.classes)
    def per_class_accuracies(self) -> List[float]:
        '''
        Compute and return the per class accuracies as a list of floats between 0 and 1.
        Returns an empty list if no data is available (after resets).
        '''
        class_accuracies = []
        for i in range(len(self.classes)):
            if self.class_total[i] != 0:
                class_accuracy = self.class_correct[i] / self.class_total[i]
            else:
                class_accuracy = 0.0
            class_accuracies.append(class_accuracy)
        return class_accuracies

