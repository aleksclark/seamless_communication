# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from fairseq2.assets import get_asset_download_manager
from fairseq2.data_type import DataType
from fairseq2.device import Device
from torch import Tensor, nn


class KmeansModel(nn.Module):
    def __init__(self, kmeans_uri: str, device: Device, dtype: DataType):
        super().__init__()
        km_path = get_asset_download_manager().download_model(kmeans_uri, kmeans_uri)
        km_model = np.load(km_path)
        centroids_numpy = km_model.transpose()
        centroids = torch.from_numpy(centroids_numpy)
        self.centroids = centroids.to(device=device, dtype=dtype)
        self.centroid_norm = (self.centroids**2).sum(0, keepdims=True)

    def forward(self, x: Tensor) -> Tensor:
        dist: Tensor = (
            x.pow(2).sum(1, keepdim=True)
            - 2 * torch.matmul(x, self.centroids)
            + self.centroid_norm
        )
        return dist.argmin(dim=-1)
