import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.model_selection import StratifiedKFold

from src.data_handler.dataset import MedicalImageDataset
from src.models.cnn_3d import Basic3DCNN
from src.training.trainer import Trainer
from src.training.losses import FocalLoss

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sanity_check_manifest(df: pd.DataFrame):
    required = {"path", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest.csv missing columns: {missing}")

    # 1) label 必须是整数类 id
    if df["label"].isna().any():
        raise ValueError("manifest contains NaN labels.")

    # 强制转 int（如果你的 label 在 csv 里是 float/str，这里会帮你统一）
    df["label"] = df["label"].astype(int)

    # 2) path 必须存在
    not_exist = df[~df["path"].apply(os.path.exists)]
    if len(not_exist) > 0:
        print("Example missing files:")
        print(not_exist.head(10)[["path", "label"]])
        raise FileNotFoundError(f"{len(not_exist)} paths in manifest do not exist.")

    # 3) 同一 path 不应该重复（可选，但推荐）
    dup = df[df.duplicated("path", keep=False)]
    if len(dup) > 0:
        print("Example duplicate paths:")
        print(dup.head(10)[["path", "label"]])
        raise ValueError("Duplicate paths found in manifest.")

    return df


def make_loaders(train_df, val_df, batch_size=2, num_workers=0):
    train_ds = MedicalImageDataset(
        train_df["path"].tolist(),
        train_df["label"].tolist(),
        normalize=True,
    )
    val_ds = MedicalImageDataset(
        val_df["path"].tolist(),
        val_df["label"].tolist(),
        normalize=True,
    )

    # Sample the data based on label
    train_labels = train_df["label"].to_numpy().astype(int)
    class_counts = np.bincount(train_labels, minlength=int(train_labels.max()) + 1)
    class_weights = 1.0 / np.maximum(class_counts, 1)          # 每类权重 ~ 1/count
    sample_weights = class_weights[train_labels]               

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),   # 每个 epoch 采样次数，在这里保持 epoch 长度不变，如果效果还是不好可以考虑延长epoch长度，这样可以让minority sample更多次
        replacement=True                   # 允许重复抽一个sample
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, 
        sampler = sampler, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


def _compute_class_weights(train_labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.
    weights[c] ∝ 1 / count[c], normalized so mean(weight)=1
    """
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0  # avoid div-by-zero (shouldn't happen with stratified splits, but safe)
    w = 1.0 / counts
    w = w * (num_classes / w.sum())  # normalize
    return torch.tensor(w, dtype=torch.float32)


def plot_confusion_matrix(cm, class_names=None, normalize=False, title='Confusion Matrix'):
    """
    Plot confusion matrix as a figure.
    
    Args:
        cm: Confusion matrix (numpy array)
        class_names: List of class names (optional)
        normalize: Whether to normalize the confusion matrix
        title: Title of the plot
    
    Returns:
        matplotlib figure
    """
    if normalize:
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm = cm.astype('float') / row_sums
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=ax, 
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    return fig

def main():
    # ====== config ======
    seed = 42
    n_splits = 5

    max_epochs = 35
    batch_size = 5
    lr = 1e-4
    weight_decay = 1e-2

    # Focal Loss parameters
    focal_gamma = 2.0              # Focusing parameter for Focal Loss (higher = more focus on hard examples)
    use_class_weights = True       # Whether to use class weights in Focal Loss

    # early stopping + scheduler
    use_early_stopping = False
    patience = 8               
    min_delta = 1e-4              # val loss 至少下降多少才算提升
    save_by = "val_loss"          # 根据val loss来储存checkpoint

    manifest_path = "/data1/yxinwang/yarong/project/src/data_handler/manifest.csv"
    ckpt_root = "checkpoints"

    # Tensor board log root
    tb_root = "runs"  # tensorboard --logdir runs
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

    # ====== setup ======
    set_seed(seed)
    device = get_device()
    print("Device:", device)
    os.makedirs(ckpt_root, exist_ok=True)
    os.makedirs(tb_root, exist_ok=True)

    # ====== 1) load manifest ======
    df = pd.read_csv(manifest_path)
    df = sanity_check_manifest(df)

    labels = df["label"].to_numpy().astype(int)
    if labels.min() < 0:
        raise ValueError("Labels must be >= 0.")

    num_classes = int(labels.max()) + 1
    print(f"Total samples: {len(df)} | num_classes: {num_classes}")
    print("Global class counts:", df["label"].value_counts().sort_index().to_dict())

    # ====== 2) Stratified K-fold ======
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df["path"].to_numpy(), labels), start=1):
        print("\n" + "=" * 60)
        print(f"Fold {fold}/{n_splits}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)

        assert set(train_df["path"]).isdisjoint(set(val_df["path"])), "Train/val overlap detected!"

        train_counts = train_df["label"].value_counts().sort_index().to_dict()
        val_counts = val_df["label"].value_counts().sort_index().to_dict()
        print("Train class counts:", train_counts)
        print("Val   class counts:", val_counts)

        train_loader, val_loader = make_loaders(
            train_df, val_df, batch_size=batch_size, num_workers=0
        )

        tb_logdir = os.path.join(tb_root, run_name, f"fold_{fold}")

        # Write to tensorboard
        writer = SummaryWriter(log_dir=tb_logdir)
        writer.add_text("meta/hparams",
                        f"seed={seed}, n_splits={n_splits}, batch_size={batch_size}, "
                        f"lr={lr}, weight_decay={weight_decay}, max_epochs={max_epochs}, "
                        f"patience={patience}, min_delta={min_delta}, save_by={save_by}, "
                        f"use_early_stopping={use_early_stopping}, "
                        f"loss=FocalLoss, focal_gamma={focal_gamma}, use_class_weights={use_class_weights}",
                        global_step=0)
        writer.add_text("data/train_class_counts", str(train_counts), global_step=0)
        writer.add_text("data/val_class_counts", str(val_counts), global_step=0)

        # ====== 3) model/optim/criterion/trainer ======
        model = Basic3DCNN(num_classes=num_classes).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Focal Loss 
        class_w = None
        if use_class_weights:
            class_w = _compute_class_weights(train_df["label"].to_numpy().astype(int), num_classes).to(device)
        criterion = FocalLoss(gamma=focal_gamma, reduction='mean') #先不叠加class weight 看看结果

        # scheduler: val loss 无改善则降 lr
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, threshold=1e-4
        )

        # 兼容不同 Trainer 写法：能传 criterion 就传；不能传就 setattr
        try:
            trainer = Trainer(model=model, optimizer=optimizer, device=device, num_classes=num_classes, criterion=criterion)
        except TypeError:
            trainer = Trainer(model=model, optimizer=optimizer, device=device, num_classes=num_classes)
            setattr(trainer, "criterion", criterion)

        # ====== 4) train loop w/ early stopping ======
        best_metric = float("inf") if save_by == "val_loss" else -1.0
        best_epoch = -1
        no_improve = 0

        best_path = os.path.join(ckpt_root, f"fold_{fold}_best.pt")

        for epoch in range(1, max_epochs + 1):
            train_stats = trainer.train_one_epoch(train_loader)
            val_stats = trainer.validate(val_loader)

            # 默认 Trainer 返回 {'loss':..., 'acc':...}
            tr_loss, tr_acc, tr_bal_acc, tr_y_pred = train_stats.get("loss"), train_stats.get("acc"), train_stats.get("bal_acc"), train_stats.get("y_pred")
            va_loss, va_acc, va_bal_acc, va_y_pred , va_y_true = val_stats.get("loss"), val_stats.get("acc"), val_stats.get("bal_acc"), val_stats.get("y_pred"), val_stats.get("y_t")
            tr_cm = train_stats.get("confusion_matrix")
            va_cm = val_stats.get("confusion_matrix")
            cur_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Fold {fold} | Epoch {epoch:02d} | lr {cur_lr:.2e} | "
                f"train loss {tr_loss:.4f} acc {tr_acc:.4f} bal_acc {tr_bal_acc:.4f} y_pred {tr_y_pred}| "
                f"val loss {va_loss:.4f} val_acc {va_acc:.4f} val_bal_acc {va_bal_acc:.4f} val_y_pred {va_y_pred} val_y_true{va_y_true}"
            )
            
            # Print confusion matrix
            if va_cm is not None:
                print(f"\nValidation Confusion Matrix (Epoch {epoch}):")
                print(va_cm)
                print()  # Empty line for readability

            # Write to tensorboard
            if tr_loss is not None:
                writer.add_scalar("train/loss", float(tr_loss), epoch)
            if tr_bal_acc is not None:
                writer.add_scalar("train/bal_acc", float(tr_bal_acc), epoch)
            if va_loss is not None:
                writer.add_scalar("val/loss", float(va_loss), epoch)
            if va_bal_acc is not None:
                writer.add_scalar("val/bal_acc", float(va_bal_acc), epoch)

            writer.add_scalar("optim/lr", float(cur_lr), epoch)
            
            # Visualize confusion matrix in tensorboard
            if va_cm is not None:
                class_names = [f"Class {i}" for i in range(num_classes)]
                # Normalized confusion matrix
                fig_norm = plot_confusion_matrix(va_cm, class_names=class_names, 
                                                normalize=True, title=f'Validation Confusion Matrix (Normalized) - Epoch {epoch}')
                writer.add_figure("confusion_matrix/val_normalized", fig_norm, epoch)
                plt.close(fig_norm)
                
                # Raw confusion matrix
                fig_raw = plot_confusion_matrix(va_cm, class_names=class_names, 
                                               normalize=False, title=f'Validation Confusion Matrix (Raw) - Epoch {epoch}')
                writer.add_figure("confusion_matrix/val_raw", fig_raw, epoch)
                plt.close(fig_raw)
            
            if tr_cm is not None:
                class_names = [f"Class {i}" for i in range(num_classes)]
                # Normalized confusion matrix for training
                fig_norm = plot_confusion_matrix(tr_cm, class_names=class_names, 
                                                normalize=True, title=f'Train Confusion Matrix (Normalized) - Epoch {epoch}')
                writer.add_figure("confusion_matrix/train_normalized", fig_norm, epoch)
                plt.close(fig_norm)

            # scheduler step on val loss
            scheduler.step(va_loss)

            # early stopping + best checkpoint (by val loss)
            improved = (best_metric - va_loss) > min_delta if save_by == "val_loss" else (va_acc - best_metric) > 0

            if improved:
                best_metric = va_loss if save_by == "val_loss" else va_acc
                best_epoch = epoch
                no_improve = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "num_classes": num_classes,
                        "fold": fold,
                        "epoch": epoch,
                        "best_val_loss": best_metric if save_by == "val_loss" else None,
                        "best_val_acc": best_metric if save_by != "val_loss" else None,
                        "manifest": manifest_path,
                        "class_weights": class_w.detach().cpu().numpy().tolist() if class_w is not None else None,
                        "focal_gamma": focal_gamma,
                        "use_class_weights": use_class_weights,
                    },
                    best_path
                )
                # Write to tensorboard
                if save_by == "val_loss":
                    writer.add_scalar("best/val_loss", float(best_metric), epoch)
                else:
                    writer.add_scalar("best/val_acc", float(best_metric), epoch)
            else:
                if use_early_stopping:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"🛑 Early stop at epoch {epoch} (best epoch {best_epoch}, best {save_by}={best_metric:.4f})")
                        break

        # 用 best checkpoint 的指标汇总（这里保存的是 best_val_loss）
        fold_results.append({"fold": fold, "best_epoch": best_epoch, "best_val_loss": best_metric, "ckpt": best_path})
        print(f"✅ Fold {fold} best at epoch {best_epoch} | best val loss: {best_metric:.4f} | saved: {best_path}")

        # Close tensor board
        writer.close()
    # ====== 5) summary ======
    print("\n" + "=" * 60)
    losses = [r["best_val_loss"] for r in fold_results]
    print("CV results:", fold_results)
    print(f"Mean best val loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")


if __name__ == "__main__":
    main()

