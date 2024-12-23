from ganslate.engines.base import BaseEngineWithInference
from ganslate.utils.metrics.val_test_metrics import ValTestMetrics
from ganslate.utils import environment
from ganslate.utils.builders import build_gan, build_loader
from ganslate.utils.trackers.validation_testing import ValTestTracker

import matplotlib.pyplot as plt
import torch


class BaseValTestEngine(BaseEngineWithInference):

    def __init__(self, conf):
        super().__init__(conf)

        self.data_loaders = build_loader(self.conf)
        # Val and test modes allow multiple datasets, this handles when it's a single dataset
        if not isinstance(self.data_loaders, dict):
            # No name needed when it's a single dataloader
            self.data_loaders = {None: self.data_loaders}
        self.current_data_loader = None

        self.tracker = ValTestTracker(self.conf)
        self.metricizer = ValTestMetrics(self.conf)
        self.visuals = {}


    ###################
    # def create_heatmap(self, real_B, fake_B):
    #     fill_image = torch.zeros_like(real_B, device=real_B.device)
    #     comp_image = torch.cat((fake_B, fill_image, real_B), dim=1)
    #     return comp_image
    # def create_heatmap(self, real_B, fake_B):
    #     difference = (fake_B - real_B) / 2.0
    #     red = torch.clamp(difference, 0, 1)
    #     blue = torch.clamp(-difference, 0, 1)
    #     green = 1.0 - torch.abs(difference)
    #     composition_image = torch.cat((red,green,blue), dim=1)
    #     return composition_image
    def create_heatmap(self, real_B, fake_B):
        difference = (fake_B - real_B) / 2.0

        # Create masks
        mask_red = difference > 0
        mask_blue = difference < 0
        mask_white = difference == 0

        # Initialize color channels
        red = torch.zeros_like(difference)
        green = torch.zeros_like(difference)
        blue = torch.zeros_like(difference)

        # Apply colors
        red[mask_red] = 1.0      # Red where fake_B > real_B
        blue[mask_blue] = 1.0    # Blue where fake_B < real_B
        red[mask_white] = 1.0    # White where fake_B == real_B
        green[mask_white] = 1.0
        blue[mask_white] = 1.0

        # Stack channels
        composition_image = torch.cat((red, green, blue), dim=1)

        return composition_image
    ####################




    def run(self, current_idx=None):
        self.logger.info(f'{"Validation" if self.conf.mode == "val" else "Testing"} started.')

        for dataset_name, data_loader in self.data_loaders.items():
            self.current_data_loader = data_loader
            for data in self.current_data_loader:
                # Collect visuals
                device = self.model.device
                self.visuals = {}
                self.visuals["real_A"] = data["A"].to(device)
                self.visuals["fake_B"] = self.infer(self.visuals["real_A"])
                self.visuals["real_B"] = data["B"].to(device)

                # Add masks if provided
                if "masks" in data:
                    self.visuals["masks"] = data["masks"]

                # Save the output as specified in dataset`s `save` method, if implemented
                metadata = data["metadata"] if "metadata" in data else None
                self.save_generated_tensor(generated_tensor=self.visuals["fake_B"],
                                           metadata=metadata,
                                           data_loader=self.current_data_loader,
                                           idx=current_idx,
                                           dataset_name=dataset_name)

                metrics = self._calculate_metrics()
                self.tracker.add_sample(self.visuals, metrics)

            self.tracker.log_samples(current_idx, dataset_name=dataset_name)

        if self.conf.mode == "test":
            self.tracker.close()

    def _calculate_metrics(self):
        # TODO: Decide if cycle metrics also need to be scaled
        original, pred, target = self.visuals["real_A"], self.visuals["fake_B"], self.visuals["real_B"]

        # Metrics on input
        compute_over_input = getattr(self.conf[self.conf.mode].metrics, "compute_over_input", False)

        # Denormalize the data if dataset has `denormalize` method defined.
        denormalize = getattr(self.current_data_loader.dataset, "denormalize", False)
        # print('if denormalize:', denormalize)
        if denormalize:
            # print("Denormalizing data...")
            # print('in _calculate_metrics original:', original.detach().clone().min(), original.detach().clone().max())
            # print('in _calculate_metrics pred before:', pred.detach().clone().min(), pred.detach().clone().max())
            # print('in _calculate_metrics target before:', target.detach().clone().min(), target.detach().clone().max())
            pred, target = denormalize(pred.detach().clone()), denormalize(target.detach().clone())
            # print('in _calculate_metrics pred after:', pred.min(), pred.max())
            # print('in _calculate_metrics target after:', target.min(), target.max())
            if compute_over_input:
                original = denormalize(original.detach().clone())

        # Standard Metrics
        metrics = self.metricizer.get_metrics(pred, target)

        if compute_over_input:
            original_metrics = self.metricizer.get_metrics(original, target)
            metrics.update({f"Original_{k}": v for k, v in original_metrics.items()})

        # Mask Metrics
        mask_metrics = {}
        if "masks" in self.visuals:
            # Remove masks dict from the visuals
            masks_dict = self.visuals.pop("masks")
            for label, mask in masks_dict.items():
                mask = mask.to(pred.device)
                # Get metrics on translated masked images
                for name, value in self.metricizer.get_metrics(pred, target, mask=mask).items():
                    key = f"{name}_{label}"
                    mask_metrics[key] = value

                # Get metrics on original masked images
                if compute_over_input:
                    for name, value in self.metricizer.get_metrics(original, target,
                                                                   mask=mask).items():
                        key = f"Original_{name}_{label}"
                        mask_metrics[key] = value

                # Add mask to visuals for logging
                self.visuals[label] = 2. * mask - 1

        # Cycle Metrics
        cycle_metrics = {}
        compute_cycle_metrics = getattr(self.conf[self.conf.mode].metrics, "cycle_metrics", False)
        if compute_cycle_metrics:
            if "direction" not in self.model.infer.__code__.co_varnames:
                raise RuntimeError("If cycle metrics are enabled, please define"
                                   " behavior of inference with a `direction` flag in"
                                   " the model's `infer()` method")

            rec_A = self.infer(self.visuals["fake_B"], direction='BA')
            cycle_metrics = self.metricizer.get_cycle_metrics(rec_A, self.visuals["real_A"])

        metrics.update(mask_metrics)
        metrics.update(cycle_metrics)



        #################
        # print(self.visuals.keys())
        # print({k:v.shape for k,v in self.visuals.items()})
        if "real_B" in self.visuals and "fake_B" in self.visuals:
            heatmap = self.create_heatmap(self.visuals["real_B"], self.visuals["fake_B"])
            self.visuals["clean_mask"] = heatmap
        # print({k:v.shape for k,v in self.visuals.items()})
        return metrics


class Validator(BaseValTestEngine):

    def __init__(self, conf, model):
        super().__init__(conf)
        self.model = model

    def _set_mode(self):
        self.conf.mode = 'val'


class Tester(BaseValTestEngine):

    def __init__(self, conf):
        super().__init__(conf)
        environment.setup_logging_with_config(self.conf)
        self.model = build_gan(self.conf)

    def _set_mode(self):
        self.conf.mode = 'test'
