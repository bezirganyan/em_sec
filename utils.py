from pytorch_lightning import Callback


class OnKeyboardInterruptCallback(Callback):
    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
        if isinstance(exception, KeyboardInterrupt):
            raise exception
        super().on_exception(trainer, pl_module, exception)