from typing import Union
from pathlib import Path

import torch
import torch.multiprocessing
import torch.optim as optim
import typer

from model.loss import masked_scaled_mse
from model.bionic import Bionic
from trainer import Trainer
from utils.common import Device, cyan
from utils.sampler import StatefulSampler, NeighborSamplerWithWeights


class TrainerBionic(Trainer):
    def __init__(self, config: Union[Path, dict]):

        super().__init__(config)
        self.train_loaders = self._make_train_loaders()
        self.inference_loaders = self._make_inference_loaders()

    def _make_train_loaders(self):
        return [
            NeighborSamplerWithWeights(
                ad,
                sizes=[10] * self.params.gat_shapes["n_layers"],
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
                sizes=[-1] * self.params.gat_shapes["n_layers"],  # all neighbors
                batch_size=1,
                shuffle=False,
                sampler=StatefulSampler(torch.arange(len(self.index))),
            )
            for ad in self.adj
        ]

    def _init_model(self):
        model = Bionic(
            len(self.index),
            self.params.gat_shapes,
            self.params.embedding_size,
            len(self.adj),
            svd_dim=self.params.svd_dim,
        )
        model.apply(self._init_model_weights)

        # Load pretrained model
        # TODO: refactor this
        if self.params.load_pretrained_model:
            typer.echo("Loading pretrained model...")
            model.load_state_dict(torch.load(f"models/{self.params.out_name}_model.pt"))

        # Push model to device
        model.to(Device())

        optimizer = optim.Adam(model.parameters(), lr=self.params.learning_rate, weight_decay=0.0)

        return model, optimizer

    def _init_model_weights(self, model):
        if hasattr(model, "weight"):
            if self.params.initialization == "kaiming":
                torch.nn.init.kaiming_uniform_(model.weight, a=0.1)
            elif self.params.initialization == "xavier":
                torch.nn.init.xavier_uniform_(model.weight)
            else:
                raise ValueError(
                    f"The initialization scheme {self.params.initialization} \
                    provided is not supported"
                )

    def _train_step(self, rand_net_idx=None):
        """Defines training behaviour.
        """

        # Get random integers for batch.
        rand_int = StatefulSampler.step(len(self.index))
        int_splits = torch.split(rand_int, self.params.batch_size)
        batch_features = self.features

        # Initialize loaders to current batch.
        if bool(self.params.sample_size):
            batch_loaders = [self.train_loaders[i] for i in rand_net_idx]
            if isinstance(self.features, list):
                batch_features = [self.features[i] for i in rand_net_idx]

            # Subset `masks` tensor.
            mask_splits = torch.split(self.masks[:, rand_net_idx][rand_int], self.params.batch_size)

        else:
            batch_loaders = self.train_loaders
            mask_splits = torch.split(self.masks[rand_int], self.params.batch_size)
            if isinstance(self.features, list):
                batch_features = self.features

        # List of losses.
        losses = [0.0 for _ in range(len(batch_loaders))]

        # Get the data flow for each input, stored in a tuple.
        for batch_masks, node_ids, *data_flows in zip(mask_splits, int_splits, *batch_loaders):
            self.optimizer.zero_grad()
            if bool(self.params.sample_size):
                training_datasets = [self.adj[i] for i in rand_net_idx]
                output, _, _, _ = self.model(
                    training_datasets,
                    data_flows,
                    batch_features,
                    batch_masks,
                    rand_net_idxs=rand_net_idx,
                )
                curr_losses = [
                    masked_scaled_mse(
                        output, self.adj[i], self.weights[i], node_ids, batch_masks[:, j]
                    )
                    for j, i in enumerate(rand_net_idx)
                ]
            else:
                training_datasets = self.adj
                output, _, _, _ = self.model(
                    training_datasets, data_flows, batch_features, batch_masks
                )
                curr_losses = [
                    masked_scaled_mse(
                        output, self.adj[i], self.weights[i], node_ids, batch_masks[:, i]
                    )
                    for i in range(len(self.adj))
                ]

            losses = [loss + curr_loss for loss, curr_loss in zip(losses, curr_losses)]
            loss_sum = sum(curr_losses)
            loss_sum.backward()

            self.optimizer.step()

        return output, losses

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
