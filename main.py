import torch
from typing import Any
from torchmetrics import MaxMetric, MeanMetric, AveragePrecision
from torchvision import models
from pytorch_lightning import LightningModule,Trainer
from partial_labels_datamodule import PartialMLDataModule


class PartialLabels(LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 20)  # Change the final layer to match the number of classes
        self.learning_rate = learning_rate
        self.validation_step_outputs = []
        num_labels=20

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.map_val = AveragePrecision(task='multilabel', average='macro', num_labels=num_labels)
        self.map_test = AveragePrecision(task='multilabel', average='macro', num_labels=num_labels)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # for tracking best so far validation accuracy
        self.map_val_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.map_val.reset()
        self.map_val_best.reset()

    def model_step(self, batch: Any):
        images, noisy_labels, labels, indices = batch
        logits = self.forward(images)
        if logits.dim() == 1:
            logits = torch.unsqueeze(logits, 0)

        preds = torch.sigmoid(logits)

        return logits, preds

    def training_step(self, batch: Any, batch_idx: int):

        # batch should return the image, noisy labels, and ground truth labels
        images, noisy_labels, labels, indices = batch
        logits, preds = self.model_step(batch)
        # Setting lower limit, and treating -1 as 0
        noisy_labels.clamp_min_(0)
        loss = self.criterion(logits, noisy_labels)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        logits, preds = self.model_step(batch)
        images, noisy_labels, labels, indices = batch
        # Setting lower limit, and treating -1 as 0
        noisy_labels.clamp_min_(0)
        loss = self.criterion(logits, noisy_labels)

        # At end of epoch
        output = {"preds": preds, "labels": labels}
        self.validation_step_outputs.append(output)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        all_preds = torch.cat([x["preds"] for x in self.validation_step_outputs], dim=0)
        all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs], dim=0)
        map_val = self.map_val(all_preds, all_labels.long())
        self.log("val/map", map_val, prog_bar=True)
        # Log the metrics

        # Update best mAP
        self.map_val_best(map_val)
        self.log("val/map_best", self.map_val_best.compute(), prog_bar=True)

        # Update clean_rate
        self.validation_step_outputs.clear()
        self.map_val.reset()

    def test_step(self, batch: Any, batch_idx: int):
        logits, preds = self.model_step(batch)
        images, noisy_labels, labels, indices = batch
        # Setting lower limit, and treating -1 as 0
        noisy_labels.clamp_min_(0)
        loss = self.criterion(logits, noisy_labels)

        map_test = self.map_test(preds, labels.long())

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/map", map_test, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        return optimizer


# Instantiate the LightningDataModule
data_module = PartialMLDataModule()

# Instantiate the LightningModule
model = PartialLabels()

# Set the trainer with the desired configuration
trainer = Trainer(max_epochs=10, devices=1, accelerator='gpu')

# Train the model
trainer.fit(model, datamodule=data_module)

# Save the best model
torch.save(model.state_dict(), "best_model.pth")

print("Finished training")
