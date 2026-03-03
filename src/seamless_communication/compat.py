# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

from fairseq2.data import SequenceData
from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.nn import BatchLayout
from fairseq2.nn.normalization import LayerNorm, StandardLayerNorm
from torch import Tensor

LayerNormFactory = Callable[..., LayerNorm]


@dataclass
class SequenceBatch:
    """Lightweight shim replacing the old fairseq2 SequenceBatch (0.2 style)."""
    seqs: Tensor
    seqs_layout: Optional[BatchLayout] = None


def get_seqs_and_seqs_layout(
    data: SequenceData,
) -> Tuple[Tensor, Optional[BatchLayout]]:
    seqs = data["seqs"]

    if not data["is_ragged"]:
        return seqs, None

    return seqs, BatchLayout(seqs.shape[:2], data["seq_lens"], device=seqs.device)


def apply_seqs_layout(
    seqs: Tensor,
    seqs_layout: Optional[BatchLayout],
    pad_value: Any = 0,
) -> Tensor:
    if seqs_layout is None:
        return seqs

    mask = seqs_layout.position_indices >= 0

    for _ in range(seqs.ndim - mask.ndim):
        mask = mask.unsqueeze(-1)

    return seqs.where(mask.to(seqs.device), pad_value)


def trim_seqs_layout(
    layout: Optional[BatchLayout], trim_len: int
) -> Optional[BatchLayout]:
    if layout is None:
        return None

    new_seq_lens = [max(1, s - trim_len) for s in layout.seq_lens]
    new_width = layout.width - trim_len
    batch_size = len(layout.seq_lens)

    return BatchLayout(
        (batch_size, new_width),
        new_seq_lens,
        device=layout.seq_lens_pt.device,
    )


def create_standard_layer_norm(
    model_dim: int,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> StandardLayerNorm:
    return StandardLayerNorm(model_dim, bias=True, device=device, dtype=dtype)
