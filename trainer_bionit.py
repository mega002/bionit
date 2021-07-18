import torch
import typer
from transformers import AdamW

from model.bionit import Bionit
from model.loss import masked_scaled_mse
from trainer import Trainer
from utils.common import Device, cyan
from utils.sampler import StatefulSampler


class TrainerBionit(Trainer):
    def _init_model(self):
        model = Bionit(
            len(self.index),
            self.params.transformer_config,
            self.params.embedding_size,
            len(self.adj),
            self.params.batch_size,
        )

        # Load pretrained model
        if self.params.load_pretrained_model:
            typer.echo("Loading pretrained model...")
            model.load_state_dict(torch.load(f"models/{self.params.out_name}_model.pt"))

        # Push model to device
        model.to(Device())

        optimizer = AdamW(model.parameters(), lr=self.params.learning_rate)

        return model, optimizer

    def _train_step(self, rand_net_idx=None):
        """Defines training behaviour.
        """

        # Get random integers for batch.
        rand_int = StatefulSampler.step(len(self.index)).to(Device())
        int_splits = torch.split(rand_int, self.params.batch_size)
        batch_features = self.features

        # Initialize loaders to current batch.
        assert not bool(self.params.sample_size)

        batch_loaders = self.train_loaders
        mask_splits = torch.split(self.masks[rand_int].to(Device()), self.params.batch_size)
        if isinstance(self.features, list):
            batch_features = self.features

        # List of losses.
        losses = [0.0 for _ in range(len(self.adj))]

        # Get the data flow for each input, stored in a tuple.
        for batch_masks, node_ids, *data_flows in zip(mask_splits, int_splits, *batch_loaders):
            self.optimizer.zero_grad()
            assert not bool(self.params.sample_size)

            training_datasets = self.adj
            output, _, _, _ = self.model(
                training_datasets, data_flows, batch_features, batch_masks, rand_net_idxs=rand_net_idx
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

    def get_num_layers(self):
        return self.params.transformer_config["num_hidden_layers"]

