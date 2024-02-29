import os
import torch


def save_model(
    model: torch.nn.Module, epoch: int, kwargs, update_best: bool = False
) -> None:
    save_dir = os.path.join(kwargs.save_dir, "checkpoints", f"{kwargs.experiment_name}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model, os.path.join(save_dir, f"model_{epoch}.ckpt"))
    if update_best:
        torch.save(model, os.path.join(save_dir, "best.ckpt"))


def load_model(
    model: torch.nn.Module, save_path: str, epoch: int, is_best: bool = False
) -> None:
    NotImplemented


