# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import warnings
import torch
import numpy as np
from manage.DataManager import DataManager
from typing import Callable, Type
from tqdm import tqdm
from models.UNet import UNet
from models.yourUNet import YourUNet
from models.yourSegNet import YourSegNet
from utils.utils import mean_dice, convert_mask_to_rgb_image
import matplotlib.pyplot as plt


class CNNTrainTestManager(object):
    """
    Class used the train and test the given model in the parameters 
    """

    def __init__(self, model,
                 trainset: torch.utils.data.Dataset,
                 testset: torch.utils.data.Dataset,
                 loss_fn: torch.nn.Module,
                 optimizer_factory: Callable[[torch.nn.Module], torch.optim.Optimizer],
                 batch_size=1,
                 validation=None,
                 use_cuda=False):
        """
        Args:
            model: model to train
            trainset: dataset used to train the model
            testset: dataset used to test the model
            loss_fn: the loss function used
            optimizer_factory: A callable to create the optimizer. see optimizer function
            below for more details
            validation: wether to use custom validation data or let the one by default
            use_cuda: to Use the gpu to train the model
        """

        device_name = 'cuda:0' if use_cuda else 'cpu'
        if use_cuda and not torch.cuda.is_available():
            warnings.warn("CUDA is not available. Suppress this warning by passing "
                          "use_cuda=False to {}()."
                          .format(self.__class__.__name__), RuntimeWarning)
            device_name = 'cpu'

        self.device = torch.device(device_name)
        if validation is not None:
            self.data_manager = DataManager(trainset, testset, batch_size=batch_size, validation=validation)
        else:
            self.data_manager = DataManager(trainset, testset, batch_size=batch_size)
        self.loss_fn = loss_fn
        self.model = model
        self.optimizer = optimizer_factory(self.model)
        self.model = self.model.to(self.device)
        self.use_cuda = use_cuda
        self.metric_values = {}

    def train(self, num_epochs):
        """
        Train the model for num_epochs times
        Args:
            num_epochs: number times to train the model
        """
        # Initialize metrics container
        self.metric_values['train_loss'] = []
        self.metric_values['train_acc'] = []
        self.metric_values['val_loss'] = []
        self.metric_values['val_acc'] = []

        # Create pytorch's train data_loader
        train_loader = self.data_manager.get_train_set()
        # train num_epochs times
        for epoch in range(num_epochs):
            print("Epoch: {} of {}".format(epoch + 1, num_epochs))
            train_loss = 0.0
            train_acc = 0.0

            with tqdm(range(len(train_loader))) as t:
                train_losses = []
                train_accuracies = []
                for i, data in enumerate(train_loader, 0):
                    # transfer tensors to selected device
                    train_inputs, train_labels = data[0].to(self.device), data[1].to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward pass
                    train_outputs = self.model(train_inputs)
                    # computes loss using loss function loss_fn
                    loss = self.loss_fn(train_outputs, train_labels)

                    # Use autograd to compute the backward pass.
                    loss.backward()

                    # updates the weights using gradient descent
                    """
                    Way it could be done manually

                    with torch.no_grad():
                        for param in self.model.parameters():
                            param -= learning_rate * param.grad
                    """
                    self.optimizer.step()

                    # Save losses for plotting purposes
                    train_losses.append(loss.item())
                    acc = self.accuracy(train_outputs, train_labels)
                    train_accuracies.append(acc)

                    # print metrics along progress bar
                    train_loss += loss.item()
                    train_acc += acc
                    #t.set_postfix({loss='{:05.3f}'.format(train_loss / (i + 1))})
                    t.set_postfix({'Train loss': train_loss / (i + 1), "train_acc":  train_acc / (i + 1)})
                    t.update()
            # evaluate the model on validation data after each epoch
            self.metric_values['train_loss'].append(np.mean(train_losses))
            self.metric_values['train_acc'].append(np.mean(train_accuracies))
            self.evaluate_on_validation_set()

        print("Finished training.")

    def evaluate_on_validation_set(self):
        """
        function that evaluate the model on the validation set every epoch
        """
        # switch to eval mode so that layers like batchnorm's layers nor dropout's layers
        # works in eval mode instead of training mode
        self.model.eval()

        # Get validation data
        val_loader = self.data_manager.get_validation_set()
        validation_loss = 0.0
        validation_losses = []
        validation_accuracies = []

        with torch.no_grad():
            for j, val_data in enumerate(val_loader, 0):
                # transfer tensors to the selected device
                val_inputs, val_labels = val_data[0].to(self.device), val_data[1].to(self.device)

                # forward pass
                val_outputs = self.model(val_inputs)

                # compute loss function
                loss = self.loss_fn(val_outputs, val_labels)
                validation_losses.append(loss.item())
                validation_accuracies.append(self.accuracy(val_outputs, val_labels))
                validation_loss += loss.item()

        self.metric_values['val_loss'].append(np.mean(validation_losses))
        self.metric_values['val_acc'].append(np.mean(validation_accuracies))

        # displays metrics
        print('Validation loss %.3f' % (validation_loss / len(val_loader)))
        print('Validation accuracy %.3f' % (np.mean(validation_accuracies)))

        # switch back to train mode
        self.model.train()

    def accuracy(self, outputs, labels):
      """
      Computes the accuracy of the model
      Args:
          outputs: outputs predicted by the model (Tensor or tuple for Deep Supervision)
          labels: real outputs of the data [B, H, W]
      Returns:
          Accuracy of the model
      """
      # Gestion explicite des modèles avec Deep Supervision
      if isinstance(self.model, YourUNet):
          # On prend uniquement la sortie finale (outputs[0]) pour le calcul des métriques
          final_output = outputs[0] if isinstance(outputs, tuple) else outputs
          return mean_dice(final_output, labels).item()
      
      # Cas standard pour UNet/YourSegNet sans Deep Supervision
      elif isinstance(self.model, UNet) or isinstance(self.model, YourSegNet):
          return mean_dice(outputs, labels).item()
      
      # Cas des modèles de classification standard
      else:
          predicted = outputs.argmax(dim=1)
          correct = (predicted == labels).sum().item()
          return correct / labels.size(0)

    def evaluate_on_test_set(self):
        """
        Evaluate the model on the test set
        :returns;
            Accuracy of the model on the test set
        """
        test_loader = self.data_manager.get_test_set()
        accuracies = 0
        with torch.no_grad():
            for data in test_loader:
                test_inputs, test_labels = data[0].to(self.device), data[1].to(self.device)
                test_outputs = self.model(test_inputs)
                accuracies += self.accuracy(test_outputs, test_labels)
        print("Accuracy of the network on the test set: {:05.3f} %".format(100 * accuracies / len(test_loader)))

    def plot_metrics(self):
        """
        Function that plots train and validation losses and accuracies after training phase
        """
        epochs = range(1, len(self.metric_values['train_loss']) + 1)

        f = plt.figure(figsize=(10, 5))
        ax1 = f.add_subplot(121)
        ax2 = f.add_subplot(122)

        # loss plot
        ax1.plot(epochs, self.metric_values['train_loss'], '-o', label='Training loss')
        ax1.plot(epochs, self.metric_values['val_loss'], '-o', label='Validation loss')
        ax1.set_title('Training and validation loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # accuracy plot
        ax2.plot(epochs, self.metric_values['train_acc'], '-o', label='Training accuracy')
        ax2.plot(epochs, self.metric_values['val_acc'], '-o', label='Validation accuracy')
        ax2.set_title('Training and validation accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()
        f.savefig('fig1.png')
        plt.show()

    def plot_image_mask_prediction(self, num_samples=3):
      """
      Function that plots an image its corresponding ground truth and the predicted mask
      Args:
          num_samples: number of samples to visualize
      """
      self.model.eval()
      
      # Get validation loader
      val_loader = self.data_manager.get_validation_set()
      
      with torch.no_grad():
          for i, (img, gt) in enumerate(val_loader):
              if i >= num_samples:
                  break
                  
              # Move data to device
              img = img.to(self.device)
              
              # Get prediction (handle deep supervision if needed)
              pred = self.model(img)
              if isinstance(pred, tuple):  # If model returns tuple (deep supervision)
                  pred = pred[0]  # Take only the final output
              
              # Convert tensors to numpy arrays
              img_np = img.cpu().numpy()
              gt_np = gt.cpu().numpy()
              pred_np = pred.argmax(dim=1).cpu().numpy()
              
              # Select first image from batch and remove channel dim if needed
              img_idx = 0  # First image in batch
              if len(img_np.shape) == 4:  # Batch format [B,C,H,W]
                  img_disp = img_np[img_idx, 0] if img_np.shape[1] == 1 else img_np[img_idx].transpose(1,2,0)
              else:
                  img_disp = img_np[0] if img_np.shape[0] == 1 else img_np.transpose(1,2,0)
              
              gt_disp = gt_np[img_idx] if len(gt_np.shape) == 3 else gt_np
              pred_disp = pred_np[img_idx] if len(pred_np.shape) == 3 else pred_np
              
              # Plot the results
              f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
              
              # Input image
              if len(img_disp.shape) == 2:
                  ax1.imshow(img_disp, cmap='gray')
              else:
                  ax1.imshow(img_disp)
              ax1.set_title('Input Image')
              ax1.axis('off')
              
              # Ground truth
              ax2.imshow(gt_disp, vmin=0, vmax=self.model.num_classes-1, cmap='jet')
              ax2.set_title('Ground Truth')
              ax2.axis('off')
              
              # Prediction
              ax3.imshow(pred_disp, vmin=0, vmax=self.model.num_classes-1, cmap='jet')
              ax3.set_title('Predicted Mask')
              ax3.axis('off')
              
              f.savefig('fig{}.png'.format(i+2))
              plt.tight_layout()
              plt.show()
      
      self.model.train()


def optimizer_setup(optimizer_class: Type[torch.optim.Optimizer], **hyperparameters) -> \
        Callable[[torch.nn.Module], torch.optim.Optimizer]:
    """
    Creates a factory method that can instanciate optimizer_class with the given
    hyperparameters.

    Why this? torch.optim.Optimizer takes the model's parameters as an argument.
    Thus we cannot pass an Optimizer to the CNNBase constructor.

    Args:
        optimizer_class: optimizer used to train the model
        **hyperparameters: hyperparameters for the model
        Returns:
            function to setup the optimizer
    """

    def f(model):
        return optimizer_class(model.parameters(), **hyperparameters)

    return f
