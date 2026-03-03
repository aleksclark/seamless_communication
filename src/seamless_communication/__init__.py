# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from pathlib import Path

from fairseq2.composition.assets import register_file_assets
from fairseq2.runtime.dependency import get_dependency_resolver

__version__ = "0.1.0"


def _register_cards() -> None:
    cards_dir = Path(__file__).parent.joinpath("cards")
    container = get_dependency_resolver()
    register_file_assets(container, cards_dir)


_register_cards()
