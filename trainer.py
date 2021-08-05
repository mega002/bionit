import math
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import typer

from utils.common import extend_path, cyan, magenta, Device
from utils.config_parser import ConfigParser
from utils.plotter import plot_losses
from utils.preprocessor import Preprocessor
from utils.sampler import StatefulSampler, NeighborSamplerWithWeights


class Trainer(ABC):
    def __init__(self, config: Union[Path, dict]):
        """Defines the relevant training and forward pass logic for BIONIC.

        A model is trained by calling `train()` and the resulting gene embeddings are
        obtained by calling `forward()`.

        Args:
            config (Union[Path, dict]): Path to config file or dictionary containing config
                parameters.
        """

        typer.secho("Using CUDA", fg=typer.colors.GREEN) if Device() == "cuda" else typer.secho(
            "Using CPU", fg=typer.colors.RED
        )

        self.params = self._parse_config(
            config
        )  # parse configuration and load into `params` namespace

        self._preprocessor = Preprocessor(
            self.params.names, delimiter=self.params.delimiter, svd_dim=self.params.svd_dim,
        )

        self.writer = (
            self._init_tensorboard()
        )  # create `SummaryWriter` for tensorboard visualization
        self.index, self.masks, self.weights, self.features, self.adj = self._preprocess_inputs()
        self.model, self.optimizer = self._init_model()

        self.train_loaders = self._make_train_loaders()
        self.inference_loaders = self._make_inference_loaders()

    def _parse_config(self, config):
        cp = ConfigParser(config)
        return cp.parse()

    def _init_tensorboard(self):
        if self.params.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            return SummaryWriter(flush_secs=10)
        return None

    def _preprocess_inputs(self):
        return self._preprocessor.process()

    @abstractmethod
    def _init_model(self):
        raise NotImplementedError()

    def _make_train_loaders(self):
        return [
            NeighborSamplerWithWeights(
                ad,
                sizes=[5] * self.get_num_layers(),
                batch_size=self.params.batch_size,
                shuffle=False,
                sampler=StatefulSampler(torch.arange(len(self.index))),
            )
            for ad in self.adj
        ]

    def _make_inference_loaders(self):
        return [
            NeighborSamplerWithWeights(
                ad,
                sizes=[-1] * self.get_num_layers(),  # all neighbors
                batch_size=1,
                shuffle=False,
                sampler=StatefulSampler(torch.arange(len(self.index))),
            )
            for ad in self.adj
        ]

    def train(self, verbosity: Optional[int] = 1):
        """Trains BIONIC model.

        TODO: this should be refactored

        Args:
            verbosity (int): 0 to supress printing (except for progress bar), 1 for regular printing.
        """

        # Track losses per epoch.
        train_loss = []

        best_loss = None
        best_state = None

        # Train model.
        for epoch in range(self.params.epochs):

            time_start = time.time()

            # Track average loss across batches.
            epoch_losses = np.zeros(len(self.adj))

            if bool(self.params.sample_size):
                rand_net_idxs = np.random.permutation(len(self.adj))
                idx_split = np.array_split(
                    rand_net_idxs, math.floor(len(self.adj) / self.params.sample_size)
                )
                for rand_idxs in idx_split:
                    _, losses = self._train_step(rand_idxs)
                    for idx, loss in zip(rand_idxs, losses):
                        epoch_losses[idx] += loss

            else:
                _, losses = self._train_step()

                epoch_losses = [
                    ep_loss + b_loss.item() / (len(self.index) / self.params.batch_size)
                    for ep_loss, b_loss in zip(epoch_losses, losses)
                ]

            if verbosity:
                progress_string = self._create_progress_string(epoch, epoch_losses, time_start)
                typer.echo(progress_string)

            # Add loss data to tensorboard visualization
            if self.params.use_tensorboard:
                if len(self.adj) <= 10:
                    writer_dct = {name: loss for name, loss in zip(self.names, epoch_losses)}
                    writer_dct["Total"] = sum(epoch_losses)
                    self.writer.add_scalars("Reconstruction Errors", writer_dct, epoch)

                else:
                    self.writer.add_scalar("Total Reconstruction Error", sum(epoch_losses), epoch)

            train_loss.append(epoch_losses)

            # Store best parameter set
            if not best_loss or sum(epoch_losses) < best_loss:
                best_loss = sum(epoch_losses)
                state = {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "best_loss": best_loss,
                }
                best_state = state
                # torch.save(state, f'checkpoints/{self.params.out_name}_model.pt')

        if self.params.use_tensorboard:
            self.writer.close()

        self.train_loss, self.best_state = train_loss, best_state

    def _create_progress_string(
        self, epoch: int, epoch_losses: List[float], time_start: float
    ) -> str:
        """Creates a training progress string to display.
        """
        sep = magenta("|")

        progress_string = (
            f"{cyan('Epoch')}: {epoch + 1} {sep} "
            f"{cyan('Loss Total')}: {sum(epoch_losses):.6f} {sep} "
        )
        if len(self.adj) <= 10:
            for i, loss in enumerate(epoch_losses):
                progress_string += f"{cyan(f'Loss {i + 1}')}: {loss:.6f} {sep} "
        progress_string += f"{cyan('Time (s)')}: {time.time() - time_start:.4f}"
        return progress_string

    def forward(self, verbosity: Optional[int] = 1):
        """Runs the forward pass on the trained BIONIC model.

        TODO: this should be refactored

        Args:
            verbosity (int): 0 to supress printing (except for progress bar), 1 for regular printing.
        """
        # Begin inference
        self.model.load_state_dict(
            self.best_state["state_dict"]
        )  # Recover model with lowest reconstruction loss
        if verbosity:
            typer.echo(
                (
                    f"""Loaded best model from epoch {magenta(f"{self.best_state['epoch']}")} """
                    f"""with loss {magenta(f"{self.best_state['best_loss']:.6f}")}"""
                )
            )

        self.model.eval()
        StatefulSampler.step(len(self.index), random=False)

        emb_list, learned_scales = self._build_embeddings()
        emb = np.concatenate(emb_list)
        emb_df = pd.DataFrame(emb, index=self.index)
        emb_df.to_csv(extend_path(self.params.out_name, "_features.tsv"), sep="\t")
        # emb_df.to_csv(extend_path(self.params.out_name, "_features.csv"))

        # Free memory (necessary for sequential runs)
        if Device() == "cuda":
            torch.cuda.empty_cache()

        # Create visualization of integrated features using tensorboard projector
        if self.params.use_tensorboard:
            self.writer.add_embedding(emb, metadata=self.index)

        # Output loss plot
        if self.params.plot_loss:
            if verbosity:
                typer.echo("Plotting loss...")
            plot_losses(
                self.train_loss, self.params.names, extend_path(self.params.out_name, "_loss.png")
            )

        # Save model
        if self.params.save_model:
            if verbosity:
                typer.echo("Saving model...")
            torch.save(self.model.state_dict(), extend_path(self.params.out_name, "_model.pt"))

        # Save internal learned network scales
        if self.params.save_network_scales:
            if verbosity:
                typer.echo("Saving network scales...")
            learned_scales = pd.DataFrame(
                learned_scales.detach().cpu().numpy(), columns=self.params.names
            ).T
            learned_scales.to_csv(
                extend_path(self.params.out_name, "_network_weights.tsv"), header=False, sep="\t"
            )

        typer.echo(magenta("Complete!"))

    def _build_embeddings(self):
        # Build embedding one node at a time
        emb_list = []

        # TODO: add verbosity control
        with typer.progressbar(
                zip(self.masks, self.index, *self.inference_loaders),
                label=f"{cyan('Forward Pass')}:",
                length=len(self.index),
        ) as progress:
            for mask, idx, *data_flows in progress:
                mask = mask.reshape((1, -1))
                dot, emb, _, learned_scales = self.model(
                    self.adj, data_flows, self.features, mask, evaluate=True
                )
                emb_list.append(emb.detach().cpu().numpy())
        return emb_list, learned_scales

    @abstractmethod
    def get_num_layers(self):
        raise NotImplementedError
