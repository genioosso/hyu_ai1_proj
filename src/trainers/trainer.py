from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from models.base_model import BaseModel


@dataclass
class TrainingHistory:

    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    validation_accuracy: list[float] = field(default_factory=list)
    validation_precision: list[float] = field(default_factory=list)
    validation_recall: list[float] = field(default_factory=list)


class Trainer:

    def __init__(
        self,
        model: BaseModel,
        device: torch.device,
        learning_rate: float,
        checkpoint_dir: Path,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        weights = class_weights.to(device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self, train_loader: DataLoader, validation_loader: DataLoader, epochs: int) -> TrainingHistory:
        history = TrainingHistory()
        best_accuracy = -1.0
        for epoch in range(epochs):
            train_metrics = self._train_one_epoch(train_loader)
            history.train_loss.append(train_metrics["loss"])
            history.train_accuracy.append(train_metrics["accuracy"])
            validation = self.validate(validation_loader)
            history.validation_loss.append(validation["loss"])
            history.validation_accuracy.append(validation["accuracy"])
            history.validation_precision.append(validation["precision"])
            history.validation_recall.append(validation["recall"])
            if validation["accuracy"] > best_accuracy:
                # 2026.05.25, scyang, 마지막 모델 말고 검증 제일 좋던 모델 저장함
                best_accuracy = validation["accuracy"]
                self.save_checkpoint(self.checkpoint_dir / "best.pt", epoch=epoch, metric=best_accuracy)
        return history

    def validate(self, data_loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        predictions: list[int] = []
        targets: list[int] = []
        with torch.no_grad():
            for batch in data_loader:
                images, layouts, labels = self._move_batch(batch)
                logits = self.model(images, layouts)
                total_loss += float(self.criterion(logits, labels).item())
                predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
                targets.extend(labels.cpu().tolist())
        return self._metric_summary(predictions, targets, total_loss / max(len(data_loader), 1))

    def save_checkpoint(self, path: Path, epoch: int, metric: float) -> None:
        torch.save({"model_state": self.model.state_dict(), "epoch": epoch, "metric": metric}, path)

    def load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])

    def _train_one_epoch(self, data_loader: DataLoader) -> dict[str, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch in data_loader:
            images, layouts, labels = self._move_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(images, layouts)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += float(loss.item())
            predictions = torch.argmax(logits.detach(), dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
        return {"loss": running_loss / max(len(data_loader), 1), "accuracy": correct / max(total, 1)}

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return batch["image"].to(self.device), batch["layout"].to(self.device), batch["label"].to(self.device)

    def _metric_summary(self, predictions: list[int], targets: list[int], loss: float) -> dict[str, float]:
        total = max(len(targets), 1)
        correct = sum(int(pred == target) for pred, target in zip(predictions, targets))
        # 2026.06.02, scyang, 지표 계산할 때 라벨 1을 좋은 화면으로 봄
        true_positive = sum(int(pred == target == 1) for pred, target in zip(predictions, targets))
        false_positive = sum(int(pred == 1 and target != 1) for pred, target in zip(predictions, targets))
        false_negative = sum(int(pred != 1 and target == 1) for pred, target in zip(predictions, targets))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        return {"loss": loss, "accuracy": correct / total, "precision": precision, "recall": recall}
