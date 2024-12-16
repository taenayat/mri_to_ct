from loguru import logger
from pathlib import Path

from ganslate.utils import communication
from ganslate.utils.trackers.base import BaseTracker
from ganslate.utils.trackers.utils import process_visuals_for_logging
import matplotlib.pyplot as plt


class TrainingTracker(BaseTracker):

    def __init__(self, conf):
        super().__init__(conf)
        self.logger = logger
        self.log_freq = conf.train.logging.freq

    def log_iter(self, learning_rates, losses, visuals, metrics):
        """Parameters: # TODO: update this
            iters (int) -- current training iteration
            losses (tuple/list) -- training losses
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        if self.iter_idx % self.log_freq != 0:
            return

        def parse_visuals(visuals):
            # Note: Gather not necessary as in val/test, enough to log one example when training.
            visuals = {k: v for k, v in visuals.items() if v is not None}
            visuals = process_visuals_for_logging(self.conf, visuals, single_example=True)
            # `single_example=True` returns a single example from the batch, selecting it
            return visuals[0]

        def parse_losses(losses):
            losses = {k: v for k, v in losses.items() if v is not None}
            # Reduce losses (avg) and send to the process of rank 0
            losses = communication.reduce(losses, average=True, all_reduce=False)
            return losses

        def parse_metrics(metrics):
            metrics = {k: v for k, v in metrics.items() if v is not None}
            # Training metrics are optional
            if metrics:
                # Reduce metrics (avg) and send to the process of rank 0
                metrics = communication.reduce(metrics, average=True, all_reduce=False)
            return metrics

        def log_message():
            message = '\n' + 20 * '-' + ' '
            # Iteration, computing time, dataloading time
            message += f"(iter: {self.iter_idx} | comp: {self.t_comp:.3f}, data: {self.t_data:.3f}"
            message += " | "
            # Learning rates
            for i, (name, learning_rate) in enumerate(learning_rates.items()):
                message += "" if i == 0 else ", "
                message += f"{name}: {learning_rate:.7f}"
            message += ') ' + 20 * '-' + '\n'
            # Losses
            for name, loss in losses.items():
                message += f"{name}: {loss:.3f} "
            self.logger.info(message)

        def log_visuals():
            self._save_image(visuals, self.iter_idx)

        visuals = parse_visuals(visuals)
        losses = parse_losses(losses)
        metrics = parse_metrics(metrics)

        log_message()
        log_visuals()

        if self.wandb:
            self.wandb.log_iter(iter_idx=self.iter_idx,
                                visuals=visuals,
                                mode="train",
                                learning_rates=learning_rates,
                                losses=losses,
                                metrics=metrics)

        if self.tensorboard:
            self.tensorboard.log_iter(iter_idx=self.iter_idx,
                                    #   visuals=visuals,
                                      visuals=None,
                                      mode="train",
                                      learning_rates=learning_rates,
                                      losses=losses,
                                      metrics=metrics,
                                      gpu_usage=True)

    def save_learning_curves(self,losses):
        import pandas as pd
        
        # losses_discriminator, losses_generator = [], []
        # for d in losses:

        #     # if 'D_A' in d:
        #     #     losses_discriminator.append(d['D_A'].detach().cpu().item())
        #     # if 'D_B' in d:
        #     #     losses_discriminator.append(d['D_B'].detach().cpu().item())
        #     # losses_discriminator.append(d['D'].detach().cpu().item())
        #     # losses_generator.append(d['G'].detach().cpu().item())

        #     losses_discriminator.append(d['D_A'].detach().cpu().item())
        #     losses_generator.append(d['G_AB'].detach().cpu().item())

            
        # plt.figure(figsize=(10, 5))
        # plt.plot(losses_discriminator, label='Discriminator', color='green', linewidth=0.5)
        # plt.plot(losses_generator, label='Generator', color='orange', linewidth=0.5)
        
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.savefig(Path(self.output_dir)/'training_curves.png')

        # # Save to csv losses of disciminator and generator together
        # losses = {'discriminator': losses_discriminator, 'generator': losses_generator}
        # losses_df = pd.DataFrame(losses)
        # losses_df.to_csv(Path(self.output_dir)/'training_losses.csv', index=False)


        # losses here is a list of dictionaries
        # now we want a dictionary of lists
        losses_detached = {}
        loss_names = list(losses[0].keys())
        if 'idt_A' in loss_names:
            loss_names.remove('idt_A')
        if 'idt_B' in loss_names:
            loss_names.remove('idt_B')

        for loss in loss_names:
            losses_detached[loss] = []

        for d in losses:
            for loss in loss_names:
                losses_detached[loss].append(d[loss].detach().cpu().item())

        losses_df = pd.DataFrame(losses_detached)
        losses_df.to_csv(Path(self.output_dir)/'training_losses.csv', index=False)

