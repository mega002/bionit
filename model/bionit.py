import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
from torch_sparse import SparseTensor
from transformers import BertConfig, BertModel

from model.layers import Interp
from utils.common import Device

from typing import Dict, List, Optional, Tuple

from utils.sampler import Adj


class Bionit(nn.Module):
    def __init__(
        self,
        in_size: int,
        transformer_config: Dict,
        emb_size: int,
        n_modalities: int,
        max_size: int
    ):
        """The BIONIC model.

        Args:
            in_size (int): Number of nodes in input networks.
            gat_shapes (Dict[str, int]): Graph attention layer hyperparameters.
            emb_size (int): Dimension of learned node features.
            n_modalities (int): Number of input networks.
            alpha (float, optional): LeakyReLU negative slope. Defaults to 0.1.
            svd_dim (int, optional): Dimension of input node feature SVD approximation.
                Defaults to 0.
        """

        super(Bionit, self).__init__()

        self.in_size = in_size
        self.emb_size = emb_size
        self.hidden_size = transformer_config["hidden_size"]
        self.n_modalities = n_modalities
        self.adj_dense_layers = []
        self.pre_gat_layers = []
        self.transformers = []
        self.post_gat_layers = []  # Dense transform after each GAT encoder.

        # Transformers
        for i in range(self.n_modalities):
            self.transformers.append(
                BertModel(
                    config=BertConfig(
                        vocab_size=self.in_size,
                        hidden_size=self.hidden_size,
                        num_hidden_layers=transformer_config["num_hidden_layers"],
                        intermediate_size=transformer_config["intermediate_size"],
                        position_embedding_type="none",
                        num_attention_heads=transformer_config["num_attention_heads"],
                        max_position_embeddings=max_size
                    )
                )
            )
            delattr(self.transformers[-1].embeddings, 'token_type_ids')

        for g, model in enumerate(self.transformers):
            model.init_weights()
            self.add_module("TRANSFORMERS_{}".format(g), model)

        # Embedding.
        self.emb = nn.Linear(self.hidden_size, emb_size)

        self.interp = Interp(self.n_modalities)

    def forward(
        self,
        datasets: List[SparseTensor],
        data_flows: List[Tuple[int, Tensor, List[Adj]]],
        features: Tensor,
        masks: Tensor,
        evaluate: Optional[bool] = False,
        rand_net_idxs: Optional[np.ndarray] = None,
    ):
        """Forward pass logic.

        Args:
            datasets (List[SparseTensor]): Input networks.
            data_flows (List[Tuple[int, Tensor, List[Adj]]]): Sampled bi-partite data flows.
                See PyTorch Geometric documentation for more details.
            features (Tensor): 2D node features tensor.
            masks (Tensor): 2D masks indicating which nodes (rows) are in which networks (columns)
            evaluate (Optional[bool], optional): Used to turn off random sampling in forward pass.
                Defaults to False.
            rand_net_idxs (Optional[np.ndarray], optional): Indices of networks if networks are being
                sampled. Defaults to None.

        Returns:
            Tensor: 2D tensor of final reconstruction to be used in loss function.
            Tensor: 2D tensor of node features. Each row is a node, each column is a feature.
            List[Tensor]: Pre-integration network-specific node feature tensors. Not currently
                implemented.
            Tensor: Learned network scaling coefficients.
        """

        if rand_net_idxs is not None:
            idxs = rand_net_idxs
        else:
            idxs = list(range(self.n_modalities))

        scales, interp_masks = self.interp(masks, idxs, evaluate)

        batch_size = data_flows[0][0]
        x_store_modality = torch.zeros(
            (batch_size, self.hidden_size), device=Device()
        )  # Tensor to store results from each modality.

        # Iterate over input networks
        for i, data_flow in enumerate(data_flows):
            net_idx = idxs[i]

            _, n_id, adjs = data_flow

            n_id = n_id.to(Device())

            transformers_output = self.transformers[net_idx](
                input_ids=n_id.unsqueeze(dim=0),
                # attention_mask=masks[node_ids, net_idx].unsqueeze(dim=0)
            )
            x = transformers_output.last_hidden_state[0]

            # take only the original ids of the batch - these will always be the first BATCH_SIZE node ids
            x = x[:batch_size]

            # batch_idx =
            x = scales[:, i] * interp_masks[:, i].reshape((-1, 1)) * x
            x_store_modality += x

        # Embedding
        emb = self.emb(x_store_modality)

        # Dot product.
        dot = torch.mm(emb, torch.t(emb))

        return dot, emb, None, None
