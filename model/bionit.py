import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
from torch_sparse import SparseTensor
from transformers import BertConfig, LongformerConfig

from model.modeling_bert import BertModel

from model.modeling_longformer import LongformerModel
from utils.common import Device

from typing import Dict, List, Optional


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
                LongformerModel(
                    config=LongformerConfig(
                        vocab_size=self.in_size,
                        hidden_size=self.hidden_size,
                        num_hidden_layers=transformer_config["num_hidden_layers"],
                        intermediate_size=transformer_config["intermediate_size"],
                        position_embedding_type="none",
                        num_attention_heads=transformer_config["num_attention_heads"]
                    )
                )
            )

        for g, model in enumerate(self.transformers):
            model.init_weights()
            self.add_module("TRANSFORMERS_{}".format(g), model)

        # Embedding.
        self.emb = nn.Linear(self.hidden_size, emb_size)

    def forward(
        self,
        datasets: List[SparseTensor],
        features: Tensor,
        masks: Tensor,
        node_ids: Tensor,
        evaluate: Optional[bool] = False,
        rand_net_idxs: Optional[np.ndarray] = None,
    ):
        """Forward pass logic.

        Args:
            datasets (List[SparseTensor]): Input networks.
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

        batch_size = node_ids.size(0)
        x_store_modality = torch.zeros(
            (batch_size, self.hidden_size), device=Device()
        )  # Tensor to store results from each modality.

        # Iterate over input networks
        for net_idx in range(self.n_modalities):
            transformers_output = self.transformers[net_idx](
                input_ids=node_ids.unsqueeze(dim=0),
                attention_mask=masks[node_ids, net_idx].unsqueeze(dim=0)
            )
            x = transformers_output.last_hidden_state[0]
            x_store_modality += x

        # Embedding
        emb = self.emb(x_store_modality)

        # Dot product.
        dot = torch.mm(emb, torch.t(emb))

        return dot, emb, None, None
