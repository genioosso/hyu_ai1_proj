from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from evaluators.evaluator import EvaluationResult
from trainers.trainer import TrainingHistory


class ResultPlotter:

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_history(self, model_name: str, history: TrainingHistory) -> None:
        self._line_plot(model_name, "loss", {"train": history.train_loss, "validation": history.validation_loss})
        self._line_plot(model_name, "accuracy", {"validation": history.validation_accuracy})
        self._line_plot(model_name, "precision", {"validation": history.validation_precision})
        self._line_plot(model_name, "recall", {"validation": history.validation_recall})

    def plot_confusion_matrix(self, model_name: str, result: EvaluationResult) -> None:
        fig, axis = plt.subplots(figsize=(4, 4))
        axis.imshow(result.confusion_matrix, cmap="Blues")
        axis.set_title(f"{model_name} Confusion Matrix")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        for row_index, row in enumerate(result.confusion_matrix):
            for column_index, value in enumerate(row):
                axis.text(column_index, row_index, str(value), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(self.output_dir / f"{model_name}_confusion_matrix.png")
        plt.close(fig)

    def _line_plot(self, model_name: str, metric: str, series: dict[str, list[float]]) -> None:
        fig, axis = plt.subplots(figsize=(6, 4))
        for label, values in series.items():
            axis.plot(range(1, len(values) + 1), values, label=label)
        axis.set_title(f"{model_name} {metric.title()}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric.title())
        axis.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / f"{model_name}_{metric}.png")
        plt.close(fig)
