# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Optional, Union, final

from fairseq2.assets import (
    AssetDownloadManager,
    AssetStore,
    get_asset_store,
    get_asset_download_manager,
)
from fairseq2.assets.card import AssetCard
from fairseq2.data.tokenizers.sentencepiece import BasicSentencePieceTokenizer as SentencePieceTokenizer
from fairseq2.data.tokenizers.sentencepiece import SentencePieceEncoder
from fairseq2.device import Device
from typing_extensions import override


@final
class CharTokenizer(SentencePieceTokenizer):
    """A character-level tokenizer used during non-autoregressive T2U decoding."""

    def __init__(self, path: Path) -> None:
        """
        :param pathname:
            The pathname of the SentencePiece model file.
        """
        super().__init__(path)

    @override
    def create_encoder(
        self,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        mode: Optional[str] = None,
        device: Optional[Device] = None,
        pin_memory: bool = False,
    ) -> SentencePieceEncoder:
        """Creates a character level encoder."""
        return SentencePieceEncoder(self.model, device=device, pin_memory=pin_memory)


class UnitYCharTokenizerLoader:
    """Loads character-level tokenizers of UnitY models."""

    def __init__(
        self, asset_store: AssetStore, download_manager: AssetDownloadManager
    ) -> None:
        """
        :param asset_store:
            The asset store to retrieve the model information.
        :param download_manager:
            The download manager to use.
        """
        self.asset_store = asset_store
        self.download_manager = download_manager

    def __call__(
        self,
        model_name_or_card: Union[str, AssetCard],
        force: bool = False,
        progress: bool = True,
    ) -> CharTokenizer:
        """
        :param model_name_or_card:
            The name of the model or an already loaded AssetCard
        """

        if isinstance(model_name_or_card, AssetCard):
            card = model_name_or_card
        else:
            card = self.get_asset_store().retrieve_card(model_name_or_card)

        uri = card.field("char_tokenizer").as_uri()

        pathname = self.download_manager.download_tokenizer(
            uri, card.name, force=force, progress=progress
        )

        return CharTokenizer(pathname)


load_unity_char_tokenizer = UnitYCharTokenizerLoader(get_asset_store(), get_asset_download_manager())
