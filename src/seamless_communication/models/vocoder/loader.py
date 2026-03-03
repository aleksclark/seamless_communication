# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Optional, Union

import torch
from fairseq2.assets import AssetCard, get_asset_store, get_asset_download_manager
from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.utils.uri import Uri

from seamless_communication.models.vocoder.builder import (
    VocoderConfig,
    create_vocoder_model,
    vocoder_archs,
)
from seamless_communication.models.vocoder.vocoder import Vocoder


def convert_vocoder_checkpoint(
    checkpoint: Dict[str, Any], config: VocoderConfig
) -> Dict[str, Any]:
    if (
        "model" in checkpoint
        and "code_generator.resblocks.0.convs1.0.weight_g" in checkpoint["model"]
    ):
        return checkpoint

    old_state_dict = checkpoint["generator"]
    new_state_dict = {}
    for key in old_state_dict:
        new_key = f"code_generator.{key}"
        new_state_dict[new_key] = old_state_dict[key]
    checkpoint["model"] = new_state_dict  # type: ignore
    del checkpoint["generator"]  # type: ignore
    return checkpoint


def load_vocoder_config(arch: str) -> VocoderConfig:
    """Load config for the given architecture."""
    fn: Callable[[], VocoderConfig] = vocoder_archs[arch]
    return fn()


def load_vocoder_model(
    model_name_or_card: Union[str, AssetCard],
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> Vocoder:
    """Load a Vocoder model from an asset card."""
    store = get_asset_store()
    if isinstance(model_name_or_card, AssetCard):
        card = model_name_or_card
    else:
        card = store.retrieve_card(model_name_or_card)

    arch = card.field("model_arch").as_(str)
    config = load_vocoder_config(arch)

    model = create_vocoder_model(config, device=device, dtype=dtype)

    download_manager = get_asset_download_manager()
    checkpoint_uri = card.field("checkpoint").as_(str)
    checkpoint_path = download_manager.download_model(Uri.parse(checkpoint_uri), card.name)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint = convert_vocoder_checkpoint(checkpoint, config)
    model.load_state_dict(checkpoint["model"])

    if device is not None:
        model = model.to(device)
    if dtype is not None:
        model = model.to(dtype)

    model.eval()
    return model
