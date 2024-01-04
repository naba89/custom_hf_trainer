"""Copyright: Nabarun Goswami (2023)."""

from typing import Dict, List

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl, \
    is_torch_tpu_available

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm


class AddExtraLossesToTrainerState(TrainerCallback):
    def __init__(self, extra_losses: List[str]):
        self.extra_losses = extra_losses

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.extra_losses = {k: torch.tensor(0.0).to(args.device) for k in self.extra_losses}
        return control


class CustomTrainer(Trainer):

    def __init__(self, extra_losses: List[str] = None, **kwargs):
        super().__init__(**kwargs)
        if extra_losses is not None:
            self.add_callback(AddExtraLossesToTrainerState(extra_losses))

    def compute_loss(self, model, inputs, return_outputs=False):
        if hasattr(self.control, 'extra_losses') and model.training:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

            if not isinstance(outputs, dict):
                raise ValueError("The model output should be a dictionary or ModelOutput and not a tuple or list.")
            for k, v in outputs.items():
                if k in self.control.extra_losses:
                    if v is not None:
                        if self.args.n_gpu > 1:
                            v = v.mean()
                        self.control.extra_losses[k] += v.detach() / self.args.gradient_accumulation_steps

            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        """ adapted from Trainer._maybe_log_save_evaluate to support logging extra losses
        """
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)

            if hasattr(self.control, 'extra_losses'):
                for k, v in self.control.extra_losses.items():
                    logs[k] = self._nested_gather(v).mean().item()
                    # reset the loss
                    self.control.extra_losses[k] -= self.control.extra_losses[k]

                    logs[k] = round(logs[k] / (self.state.global_step - self._globalstep_last_logged), 4)

            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
