from pytorch_lightning import Callback
from torch import nn


class OnKeyboardInterruptCallback(Callback):
    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
        if isinstance(exception, KeyboardInterrupt):
            raise exception
        super().on_exception(trainer, pl_module, exception)


def append_dropout(model, rate=0.2):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            append_dropout(module)
        if isinstance(module, nn.ReLU):
            new = nn.Sequential(module, nn.Dropout2d(p=rate, inplace=True))
            setattr(model, name, new)