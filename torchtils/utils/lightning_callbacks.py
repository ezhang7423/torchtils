import os
import pytorch_lightning as pl

class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_train_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                if self.prefix is None:
                    filename = f"{epoch=}_{global_step=}.ckpt"
                else:
                    filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


class CheckpointEveryEpoch(pl.Callback):
    """
    Save a checkpoint every epoch, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        prefix="N-Step-Checkpoint",
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.prefix = prefix

    def on_train_epoch_start(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        if self.prefix is None:
            filename = f"{epoch=}.ckpt"
        else:
            filename = f"{self.prefix}_{epoch=}.ckpt"
        ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
        trainer.save_checkpoint(ckpt_path)