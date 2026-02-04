import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score


class Trainer:
    def __init__(self, model, optimizer, device, num_classes, criterion=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.num_classes = num_classes
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

    @staticmethod
    def _compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
        """
        y_true, y_pred are 1D CPU tensors of shape [N]
        """
        yt = y_true.numpy()
        yp = y_pred.numpy()
        return {
            "acc": float((yp == yt).mean()) if len(yt) > 0 else 0.0,
            "bal_acc": float(balanced_accuracy_score(yt, yp)) if len(yt) > 0 else 0.0,
            "macro_f1": float(f1_score(yt, yp, average="macro")) if len(yt) > 0 else 0.0,
        }

    def train_one_epoch(self, loader) -> dict:
        self.model.train()

        total_loss = 0.0
        all_preds = []
        all_targets = []

        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs

            preds = logits.argmax(dim=1)
            all_preds.append(preds.detach().cpu())
            all_targets.append(y.detach().cpu())

        y_pred = torch.cat(all_preds) if all_preds else torch.empty(0, dtype=torch.long)
        y_true = torch.cat(all_targets) if all_targets else torch.empty(0, dtype=torch.long)

        out = {"loss": total_loss / max(len(y_true), 1)}
        out.update(self._compute_metrics(y_true, y_pred))
        return out

    @torch.no_grad()
    def validate(self, loader) -> dict:
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []

        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            logits = self.model(x)
            loss = self.criterion(logits, y)

            bs = x.size(0)
            total_loss += loss.item() * bs

            preds = logits.argmax(dim=1)
            all_preds.append(preds.detach().cpu())
            all_targets.append(y.detach().cpu())

        y_pred = torch.cat(all_preds) if all_preds else torch.empty(0, dtype=torch.long)
        y_true = torch.cat(all_targets) if all_targets else torch.empty(0, dtype=torch.long)

        out = {"loss": total_loss / max(len(y_true), 1)}
        out.update(self._compute_metrics(y_true, y_pred))
        return out
