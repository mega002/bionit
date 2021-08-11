import torch
import typer
from transformers import AdamW

from model.bionit import Bionit, get_edge_buckets_by_weights
from model.loss import masked_scaled_mse
from trainer import Trainer
from utils.common import Device, cyan
from utils.sampler import StatefulSampler


class TrainerBionit(Trainer):
    def _init_model(self):

        edge_buckets = None
        buckets_weights_mult = None
        if self.params.transformer_config["position_embedding_type"] == "relative_key":
            buckets_weights_mult = self.params.buckets_weights_mult
            edge_buckets = self.create_edge_buckets(buckets_weights_mult)

        model = Bionit(
            len(self.index),
            self.params.transformer_config,
            self.params.embedding_size,
            len(self.adj),
            self.params.batch_size,
            edge_buckets,
            buckets_weights_mult
        )

        # Load pretrained model
        if self.params.load_pretrained_model:
            typer.echo("Loading pretrained model...")
            model.load_state_dict(torch.load(f"models/{self.params.out_name}_model.pt"))

        # Push model to device
        model.to(Device())

        optimizer = AdamW(model.parameters(), lr=self.params.learning_rate)

        return model, optimizer

    def create_edge_buckets(self, buckets_weights_mult):
        modalities_edge_buckets = []
        for i in range(len(self.adj)):

            edge_weight = self.adj[i].edge_weight
            edge_buckets = torch.unique(get_edge_buckets_by_weights(edge_weight,
                                                                    buckets_weights_mult,
                                                                    edge_weight.device))

            # convert the buckets tensor to list for adding 0 bucket for the node pairs that aren't connected by an edge
            list_of_edge_buckets = edge_buckets.tolist()
            list_of_edge_buckets.insert(0, 0)
            edge_buckets = torch.Tensor(list_of_edge_buckets).to(edge_weight.device).int()

            modalities_edge_buckets.append(edge_buckets)

        return modalities_edge_buckets

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

