# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import Any, Callable, List, Dict, Optional, Union

import torch
from fairseq2.assets import AssetCard, get_asset_store, get_asset_download_manager
from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.utils.uri import Uri

from seamless_communication.models.aligner.builder import (
    UnitY2AlignmentConfig,
    aligner_archs,
    create_unity2_alignment_model,
)
from seamless_communication.models.aligner.model import (
    UnitY2AlignmentModel,
)
from seamless_communication.models.unity.char_tokenizer import load_unity_char_tokenizer


def convert_unity2_aligner_checkpoint(
    checkpoint: Dict[str, Any], config: UnitY2AlignmentConfig
) -> Dict[str, Any]:
    if (
        "model" in checkpoint
        and "alignment_encoder.t_conv.1.weight" in checkpoint["model"]
    ):
        return checkpoint

    alignment_frontend_statedict = {}
    text_emb_state_keymap = {"weight": "alignment_frontend.embed_text.weight"}
    for k, v in checkpoint["text_emb_state"].items():
        alignment_frontend_statedict[text_emb_state_keymap[k]] = v

    unit_emb_state_keymap = {"weight": "alignment_frontend.embed_unit.weight"}
    for k, v in checkpoint["unit_emb_state"].items():
        alignment_frontend_statedict[unit_emb_state_keymap[k]] = v

    alignment_encoder_state_dict = {}
    for k, v in checkpoint["aligner_state"].items():
        alignment_encoder_state_dict[f"alignment_encoder.{k}"] = v

    model_state = {
        **alignment_encoder_state_dict,
        **alignment_frontend_statedict,
    }

    char_embeds = model_state["alignment_frontend.embed_text.weight"]

    index_mapping = _get_char_index_mapping(config)
    vocab_size = len(index_mapping)
    char_embeds[torch.arange(vocab_size)] = char_embeds[index_mapping]

    checkpoint["model"] = model_state

    return checkpoint


def _get_char_index_mapping(config: UnitY2AlignmentConfig) -> List[int]:
    char_tokenizer = load_unity_char_tokenizer(config.model_name_or_card)
    spm_order = [
        char_tokenizer.model.index_to_token(i)
        for i in range(char_tokenizer.model.vocabulary_size)
    ][4:]
    spm_to_dict_mapping = {
        ch: idx
        for (idx, ch) in zip(
            range(4, char_tokenizer.model.vocabulary_size),
            sorted(spm_order),
        )
    }
    model_to_dict_mapping = [0, 1, 2, 3] + [spm_to_dict_mapping[ch] for ch in spm_order]
    return model_to_dict_mapping


def load_unity2_alignment_config(arch: str) -> UnitY2AlignmentConfig:
    """Load config for the given architecture."""
    fn: Callable[[], UnitY2AlignmentConfig] = aligner_archs[arch]
    return fn()


def load_unity2_alignment_model(
    model_name_or_card: Union[str, AssetCard],
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> UnitY2AlignmentModel:
    """Load a UnitY2 Alignment model from an asset card."""
    store = get_asset_store()
    if isinstance(model_name_or_card, AssetCard):
        card = model_name_or_card
    else:
        card = store.retrieve_card(model_name_or_card)

    arch = card.field("model_arch").as_(str)
    config = load_unity2_alignment_config(arch)

    model = create_unity2_alignment_model(config, device=device, dtype=dtype)

    download_manager = get_asset_download_manager()
    checkpoint_uri = card.field("checkpoint").as_(str)
    checkpoint_path = download_manager.download_model(Uri.parse(checkpoint_uri), card.name)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint = convert_unity2_aligner_checkpoint(checkpoint, config)
    model.load_state_dict(checkpoint["model"])

    if device is not None:
        model = model.to(device)
    if dtype is not None:
        model = model.to(dtype)

    model.eval()
    return model
