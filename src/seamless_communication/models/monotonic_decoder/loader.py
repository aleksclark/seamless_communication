# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Optional, Union

import torch

from fairseq2.assets import get_asset_store, get_asset_download_manager, AssetCard
from fairseq2.models.utils.checkpoint import convert_fairseq_state_dict
from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.utils.uri import Uri

from seamless_communication.models.monotonic_decoder.builder import (
    MonotonicDecoderConfig,
    create_monotonic_decoder_model,
    monotonic_decoder_archs,
)
from seamless_communication.models.monotonic_decoder.model import (
    MonotonicDecoderModel,
)


def convert_monotonic_checkpoint(
    checkpoint: Dict[str, Any], config: MonotonicDecoderConfig
) -> Dict[str, Any]:
    state_dict = checkpoint["model"]

    # Check if we have a fairseq2 checkpoint.
    if "text_decoder.layers.0.self_attn.k_proj.weight" in state_dict:
        return checkpoint

    key_map = {
        # fmt: off
        r"^decoder\.embed_tokens\.":                                            r"text_decoder_frontend.embed.",
        r"^decoder\.layers\.([0-9]+)\.self_attn\.out_proj\.":                   r"text_decoder.layers.\1.self_attn.output_proj.",
        r"^decoder\.layers\.([0-9]+)\.self_attn\.":                             r"text_decoder.layers.\1.self_attn.",
        r"^decoder\.layers\.([0-9]+)\.self_attn_layer_norm\.":                  r"text_decoder.layers.\1.self_attn_layer_norm.",
        r"^decoder\.layers\.([0-9]+)\.encoder_attn\.out_proj\.":                r"text_decoder.layers.\1.encoder_decoder_attn.output_proj.",
        r"^decoder\.layers\.([0-9]+)\.encoder_attn\.energy_bias":               r"text_decoder.layers.\1.p_choose_layer.energy_bias",
        r"^decoder\.layers\.([0-9]+)\.encoder_attn\.source_energy_layer\.":     r"text_decoder.layers.\1.p_choose_layer.k_energy_proj.",
        r"^decoder\.layers\.([0-9]+)\.encoder_attn\.target_energy_layer\.":     r"text_decoder.layers.\1.p_choose_layer.q_energy_proj.",
        r"^decoder\.layers\.([0-9]+)\.encoder_attn\.":                          r"text_decoder.layers.\1.encoder_decoder_attn.",
        r"^decoder\.layers\.([0-9]+)\.encoder_attn_layer_norm\.":               r"text_decoder.layers.\1.encoder_decoder_attn_layer_norm.",
        r"^decoder\.layers\.([0-9]+)\.fc1\.":                                   r"text_decoder.layers.\1.ffn.inner_proj.",
        r"^decoder\.layers\.([0-9]+)\.fc2\.":                                   r"text_decoder.layers.\1.ffn.output_proj.",
        r"^decoder\.layers\.([0-9]+)\.final_layer_norm\.":                      r"text_decoder.layers.\1.ffn_layer_norm.",
        r"^decoder\.layer_norm\.":                                              r"text_decoder.layer_norm.",
        r"^decoder\.output_projection\.":                                       r"final_proj.",
        # fmt: on
    }

    # Convert to fairseq2.
    checkpoint = convert_fairseq_state_dict(state_dict, key_map)
    checkpoint["model"] = state_dict
    return checkpoint

    state_dict = checkpoint["model"]

    embeds = state_dict["final_proj.weight"]

    # fairseq had a bug that accidentally introduced a dummy token in the
    # embedding table of NLLB-100. We just discard it.
    if embeds.size(0) == 256103:  # means NLLB-100
        embeds = embeds[:-1]

        state_dict["final_proj.weight"] = embeds

    # fairseq checkpoints have duplicate embedding weights. Ensure that we
    # use a single embedding table in fairseq2.
    state_dict["text_decoder_frontend.embed.weight"] = embeds

    # The embedding positions of the control symbols in fairseq's dict do
    # not match the SentencePiece model of the tokenizer.
    with torch.inference_mode():
        # (BOS, PAD, EOS, UNK) -> (PAD, UNK, BOS, EOS)
        embeds[[0, 1, 2, 3]] = embeds[[1, 3, 0, 2]]

    return checkpoint


def load_monotonic_decoder_config(arch: str) -> MonotonicDecoderConfig:
    """Load config for the given architecture."""
    fn: Callable[[], MonotonicDecoderConfig] = monotonic_decoder_archs[arch]
    return fn()


def load_monotonic_decoder_model(
    model_name_or_card: Union[str, AssetCard],
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> MonotonicDecoderModel:
    """Load a Monotonic Decoder model from an asset card."""
    store = get_asset_store()
    if isinstance(model_name_or_card, AssetCard):
        card = model_name_or_card
    else:
        card = store.retrieve_card(model_name_or_card)

    arch = card.field("model_arch").as_(str)
    config = load_monotonic_decoder_config(arch)

    model = create_monotonic_decoder_model(config, device=device, dtype=dtype)

    download_manager = get_asset_download_manager()
    checkpoint_uri = card.field("checkpoint").as_(str)
    checkpoint_path = download_manager.download_model(Uri.parse(checkpoint_uri), card.name)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint = convert_monotonic_checkpoint(checkpoint, config)
    model.load_state_dict(checkpoint["model"])

    if device is not None:
        model = model.to(device)
    if dtype is not None:
        model = model.to(dtype)

    model.eval()
    return model
