from typing import Union, Tuple, Dict

import torch
import pytorch_lightning as pl
from torch import nn
from torch import Tensor
from torch.optim import Optimizer, SGD
from torchmetrics import F1Score


class ConvBlock(nn.Module):
    """Basic Conv. Neural Network building block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]],
        padding: Union[int, Tuple[int]],
        padding_mode: str = "zeros",
        bias: bool = False,
        activation: nn.Module = nn.SiLU,
        norm_layer: nn.Module = nn.BatchNorm2d,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            bias=bias,
        )

        self.norm_layer = norm_layer
        if self.norm_layer is not None:
            self.norm_layer = self.norm_layer(out_channels)

        self.activation = activation
        if self.activation is not None:
            self.activation = activation()

    def forward(self, x: Tensor) -> Tensor:
        """Apply conv. filters, normalization and activation function."""
        out = self.conv(x)
        if self.norm_layer is not None:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class ResNetBlock(nn.Module):
    """Residual learning building block.

    Residual learning building block stacking two conv. blocks. Downsampling
    is applied at the first conv. block (if stride > 1). Similarly, the
    number of channels is expanded / reduced at the first conv. block.
    The skip connection is inspired by https://arxiv.org/pdf/2007.03898.pdf.
    It downsamples the input using 1x1 convolutions.

    Parameters
    ----------
    in_channels : int
        Number of channels of the input tensor.
    out_channels : int
        Number of channels of the output tensor. The number of channels is
        expanded / reduced at the first conv. layer.
    stride : int
        Downsampling performed in the first conv. layer.
    activation : nn.Module
        Nonlinear activation function. Activations are applied within the
        first conv. block and after the skip connection and main branch
        are accumulated.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        activation: nn.Module = nn.SiLU,
        norm_layer: nn.Module = nn.BatchNorm2d,
    ):
        super().__init__()

        self.convblock1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            activation=activation,
            norm_layer=norm_layer,
        )

        self.convblock2 = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=None,
            norm_layer=norm_layer,
        )

        self.skipblock = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            activation=None,
            norm_layer=norm_layer,
        )

        self.activation = activation()

    def forward(self, x: Tensor) -> Tensor:
        """Pass input tensor thorugh residual block."""
        skip = self.skipblock(x)
        out = self.convblock1(x)
        out = self.convblock2(out)
        out = self.activation(out + skip)

        return out


class ResNetModel(nn.Module):
    """ResNet neural network architecture for FashionMNIST.

    The architecture is a lightweight version of the ResNet proposed
    in https://arxiv.org/pdf/1512.03385.pdf. As in the paper, the architecture
    does not use strong regularization such as dropout or maxout. After an
    initial conv. layer and pooling stage, the architecture stacks ResNet
    blocks. Finally, logits for each class are computed using global average
    pooling and a fully connected layer.

    Parameters
    ----------
    num_classes : int
        Number of possible class labels in the classification task.
    activation : nn.Module
        Activation function to use throughout the architecture.
    norm_layer : nn.Module
        Normalization layer to use throughout the architecture.
    """

    def __init__(
        self,
        num_classes: int = 10,
        activation: nn.Module = nn.SiLU,
        norm_layer: nn.Module = nn.BatchNorm2d,
    ):
        super().__init__()

        self.conv5x5block_l1 = ConvBlock(
            in_channels=1,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2,
            activation=activation,
            norm_layer=norm_layer,
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.resblock_l2 = ResNetBlock(
            in_channels=32,
            out_channels=32,
            stride=2,
            activation=activation,
            norm_layer=norm_layer,
        )

        self.resblock_l3 = ResNetBlock(
            in_channels=32,
            out_channels=32,
            stride=1,
            activation=activation,
            norm_layer=norm_layer,
        )

        self.resblock_l4 = ResNetBlock(
            in_channels=32,
            out_channels=64,
            stride=2,
            activation=activation,
            norm_layer=norm_layer,
        )

        self.resblock_l5 = ResNetBlock(
            in_channels=64,
            out_channels=64,
            stride=1,
            activation=activation,
            norm_layer=norm_layer,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fclayer = nn.Linear(64, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        """Map FashionMNIST images to logits for each class.

        Parameters
        ----------
        images : Tensor (Batch size, Channels, Height, Width)
            Batch of FashionMNIST images.

        Returns
        -------
        Tensor (Batch size, Num. Classes)
            Raw, unnormalized logit scores for each class.
        """
        out = self.conv5x5block_l1(images)
        out = self.maxpool(out)

        out = self.resblock_l2(out)
        out = self.resblock_l3(out)

        out = self.resblock_l4(out)
        out = self.resblock_l5(out)

        out = self.avgpool(out)

        out = torch.flatten(out, start_dim=1)
        out = self.fclayer(out)

        return out


class ResNetFashionMnistModule(pl.LightningModule):
    """Training module for the FashionMNIST classification task.

    During training, minimizes the cross-entropy among predicted and
    ground-truth labels as an auxiliary to the F1 metric (used during
    validation & testing). Optimizes the NN with a SGD optimizer.

    Parameters
    ----------
    model : nn.Module
        Neural network that maps input images to logits for each class label.
    num_classes : int
        Number of classes of the classification task.
    lr : float
        Learning rate of the SGD optimizer.
    """

    def __init__(self, model: nn.Module, num_classes: int, lr: float = 1e-3):
        super().__init__()

        self.model = model
        self.lr = lr

        self.loss = nn.CrossEntropyLoss()
        self.f1 = F1Score(num_classes=num_classes)

    def forward(self, *args, **kwargs):
        """Wrap the forward pass of the inner model."""
        return self.model(*args, **kwargs)

    def training_step(self, batch: Tuple[Tensor], batch_idx: int) -> Dict:
        """Compute the cross-entropy between predicted and ground-truth labels.

        Parameters
        ----------
        batch : Tuple[Tensor]
            Batch of FashionMNIST images and their target labels.
        batch_idx : int
            Index of batch in training data loader.

        Returns
        -------
        Dict
            Cross-entropy training loss.
        """
        images, targets = batch

        logits = self(images)
        loss = self.loss(logits, targets)
        self.log("train_cross_entropy", loss)

        return {"loss": loss}

    def validation_step(self, batch: Tuple[Tensor], batch_idx: int) -> Dict:
        """Compute the predicted class labels.

        Parameters
        ----------
        batch : Tuple[Tensor]
            Batch of FashionMNIST images and their target labels.
        batch_idx : int
            Index of batch in training data loader.

        Returns
        -------
        Dict
            Predicted and ground-truth labels of the images.
        """
        images, targets = batch

        logits = self(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        return {"Targets": targets, "Preds.": preds}

    def validation_epoch_end(self, outputs: Dict) -> Dict:
        """Compare predicted and ground-truth class labels in terms of the F1 score.

        Parameters
        ----------
        batch : Tuple[Tensor]
            Batch of FashionMNIST images and their target labels.
        batch_idx : int
            Index of batch in training data loader.

        Returns
        -------
        Dict
            Dict that stores the F1 score of validation stage.
        """
        targets = torch.cat([output["Targets"] for output in outputs])
        preds = torch.cat([output["Preds."] for output in outputs])

        f1_score = self.f1(preds, targets)
        self.log("val_f1_score", f1_score)

        return {"val_loss": f1_score}

    def configure_optimizers(self) -> Optimizer:
        """Set up SGD optimizer."""
        return SGD(self.parameters(), lr=self.lr)
