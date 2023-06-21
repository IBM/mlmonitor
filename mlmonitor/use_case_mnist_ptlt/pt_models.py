# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torchvision import transforms
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch_utils import MNIST


# Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        # return F.softmax(x, dim=1)


class PytorchLightning_CNN_MNIST(pl.LightningModule):
    def __init__(self, arguments):
        super().__init__()
        self.save_hyperparameters()

        # Set our init args as class attributes
        self.train_data_dir = arguments.get("train")
        self.test_data_dir = arguments.get("test")
        self.hidden_size = arguments.get("hidden_size")
        self.learning_rate = arguments.get("learning_rate")
        self.batch_size = arguments.get("batch_size")
        self.test_batch_size = arguments.get("test_batch_size")
        self.epochs = arguments.get("epochs")
        self.beta_1 = arguments.get("beta_1")
        self.beta_2 = arguments.get("beta_2")
        self.weight_decay = arguments.get("weight_decay")

        use_cuda = arguments.get("num_gpus")
        self.dev = torch.device("cuda" if use_cuda > 0 else "cpu")
        self.seed = arguments.get("seed")
        # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),])
        self.loss_fn = nn.CrossEntropyLoss()

        if use_cuda:
            torch.cuda.manual_seed(arguments.get("seed"))

        self.num_classes = 10
        self.dims = (1, 28, 28)

        # Define PyTorch model
        self.model = ConvNet().to(self.dev)
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = F.cross_entropy

        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, x):
        x = self.model(x)
        return F.softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        # loss = self.loss_fn(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        # loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer = optim.Adam(
            self.parameters(),
            betas=(self.beta_1, self.beta_2),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        train_loader = DataLoader(
            MNIST(self.train_data_dir, train=True),
            batch_size=self.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            MNIST(self.test_data_dir, train=False),
            batch_size=self.test_batch_size,
            shuffle=False,
        )
        return train_loader, test_loader

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.train_data_dir, train=True)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.test_data_dir, train=False)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


class PytorchLightningMNIST(pl.LightningModule):
    def __init__(self, arguments):
        super().__init__()
        self.save_hyperparameters()

        # Set our init args as class attributes
        self.train_data_dir = arguments.get("train")
        self.test_data_dir = arguments.get("test")
        self.hidden_size = arguments.get("hidden_size")
        self.learning_rate = arguments.get("learning_rate")
        self.batch_size = arguments.get("batch_size")
        self.test_batch_size = arguments.get("test_batch_size")
        self.epochs = arguments.get("epochs")

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.num_classes),
        )

        self.accuracy = Accuracy()

    def forward(self, x):
        x = self.model(x)
        return F.softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        train_loader = DataLoader(
            MNIST(self.train_data_dir, train=True),
            batch_size=self.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            MNIST(self.test_data_dir, train=False),
            batch_size=self.test_batch_size,
            shuffle=False,
        )
        return train_loader, test_loader

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.train_data_dir, train=True)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.test_data_dir, train=False)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)
