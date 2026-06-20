from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from models.base_model import BaseModel


@dataclass(frozen=True)
class EvaluationResult:

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: list[list[int]]
    prediction_rows: list[dict[str, object]] = field(default_factory=list)


class Evaluator:

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def evaluate(self, model: BaseModel, data_loader: DataLoader) -> EvaluationResult:
        model.eval()
        predictions: list[int] = []
        targets: list[int] = []
        positive_scores: list[float] = []
        sample_ids: list[str] = []
        with torch.no_grad():
            for batch in data_loader:
                images = batch["image"].to(self.device)
                layouts = batch["layout"].to(self.device)
                labels = batch["label"].to(self.device)
                logits = model(images, layouts)
                probabilities = torch.softmax(logits, dim=1)
                predictions.extend(torch.argmax(probabilities, dim=1).cpu().tolist())
                # 2026.05.18, scyang, 사용 가능 점수만 모아두면 나중에 면적 계산됨
                positive_scores.extend(probabilities[:, min(1, probabilities.shape[1] - 1)].cpu().tolist())
                targets.extend(labels.cpu().tolist())
                sample_ids.extend(str(sample_id) for sample_id in batch.get("sample_id", []))
        return self._build_result(predictions, targets, positive_scores, sample_ids)

    def _build_result(
        self,
        predictions: list[int],
        targets: list[int],
        scores: list[float],
        sample_ids: list[str],
    ) -> EvaluationResult:
        matrix = [[0, 0], [0, 0]]
        for predicted, target in zip(predictions, targets):
            if target in (0, 1) and predicted in (0, 1):
                matrix[target][predicted] += 1
        tn, fp = matrix[0]
        fn, tp = matrix[1]
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        roc_auc = self._binary_auc(targets, scores)
        rows = [
            {
                "sample_id": sample_id,
                "true_label": target,
                "true_class": self._class_name(target),
                "predicted_label": predicted,
                "predicted_class": self._class_name(predicted),
                "usable_probability": score,
                "is_correct": int(predicted == target),
            }
            for sample_id, target, predicted, score in zip(sample_ids, targets, predictions, scores)
        ]
        return EvaluationResult(accuracy, precision, recall, f1, roc_auc, matrix, rows)

    def _class_name(self, label: int) -> str:
        return "usable" if label == 1 else "risk"

    def _binary_auc(self, targets: list[int], scores: list[float]) -> float:
        positives = [score for score, target in zip(scores, targets) if target == 1]
        negatives = [score for score, target in zip(scores, targets) if target == 0]
        if not positives or not negatives:
            return 0.0
        # 2026.05.18, scyang, 의존성 늘리기 싫어서 작은 계산은 직접 함
        wins = sum(float(pos > neg) + 0.5 * float(pos == neg) for pos in positives for neg in negatives)
        return wins / (len(positives) * len(negatives))
