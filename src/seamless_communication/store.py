# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# MIT_LICENSE file in the root directory of this source tree.

from pathlib import Path

from fairseq2.assets import get_asset_store
from fairseq2.assets.metadata_provider import load_in_memory_asset_metadata


def add_gated_assets(model_dir: Path) -> None:
    store = get_asset_store()

    if "gated" not in store._envs:
        store._envs.append("gated")

    model_dir = model_dir.resolve()

    gated_metadata = [
        {
            "name": "seamless_expressivity@gated",
            "checkpoint": str(model_dir.joinpath("m2m_expressive_unity.pt")),
        },
        {
            "name": "vocoder_pretssel@gated",
            "checkpoint": str(model_dir.joinpath("pretssel_melhifigan_wm.pt")),
        },
        {
            "name": "vocoder_pretssel_16khz@gated",
            "checkpoint": str(model_dir.joinpath("pretssel_melhifigan_wm-16khz.pt")),
        },
    ]

    provider = load_in_memory_asset_metadata("gated_assets", gated_metadata)
    store._metadata_providers.append(provider)
