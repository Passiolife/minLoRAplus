import pytorch_lightning as pl
import timm
import torch
from functools import partial
import torchmetrics
import minloraplus
from minloraplus.model import add_lora_by_name


class BaseClassifer(pl.LightningModule):

    def __init__(self, num_classes):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(self.device)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.TensorType:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(torch.argmax(y_hat, dim=1), y)
        self.log(f'train_loss', loss, prog_bar=True, on_step=True)
        self.log(f'train_acc', acc, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.TensorType:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(torch.argmax(y_hat, dim=1), y)
        self.log(f'val_loss', loss, prog_bar=True, on_step=True)
        self.log(f'val_acc', acc, prog_bar=True, on_step=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer for training.

        Returns:
        - optimizer (torch.optim.Optimizer): The optimizer object.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)
        return optimizer


class ViTClassifier(BaseClassifer):

    def __init__(self, num_classes, model_name='vit_tiny_r_s16_p8_224', lora_targets=["qkv"], img_size=224, rank=8):
        super().__init__(num_classes)
        self.model  = timm.create_model(
                                model_name,
                                img_size=img_size,
                                pretrained=True,
                                num_classes=num_classes,  # remove classifier nn.Linear
                            )
        for n, p in self.model.named_parameters():
            if "head" not in n:
                p.requires_grad = False
        config = {
            torch.nn.Linear: {
                "weight": partial(minloraplus.LoRAParametrization.from_linear, rank=rank),
            },
        }
        add_lora_by_name(self.model, lora_targets, lora_config=config)


class CNNClassifier(BaseClassifer):

    def __init__(self, num_classes, model_name='efficientnet_b0', lora_targets=["conv_pw"], img_size=224, rank=8):
        super().__init__(num_classes)
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes
        )
        for n, p in self.model.named_parameters():
            if "classifier" not in n:
                p.requires_grad = False

        config = {
            torch.nn.Conv2d: {
                "weight": partial(minloraplus.LoRAParametrization.from_conv2d, rank=rank),
            },
        }
        add_lora_by_name(self.model, lora_targets, lora_config=config)