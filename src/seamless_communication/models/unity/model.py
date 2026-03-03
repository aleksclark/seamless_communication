# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Final, Optional, Tuple, Union, final

from fairseq2.nn import BatchLayout, IncrementalStateBag, Projection
from fairseq2.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    TransformerFrontend,
)
from torch import Tensor
from torch.nn import Module

from seamless_communication.models.generator.ecapa_tdnn import ECAPA_TDNN
from seamless_communication.models.unity.fft_decoder import FeedForwardTransformer
from seamless_communication.models.unity.nar_decoder_frontend import NARDecoderFrontend

UNITY_FAMILY: Final = "unity"


@dataclass
class SequenceModelOutput:
    logits: Tensor
    pad_idx: Optional[int]


@final
class UnitYModel(Module):
    """Represents a UnitY model as described in
    :cite:t`https://doi.org/10.48550/arxiv.2212.08055`.

    Note that this implementation is augmented with a text encoder to enable
    translating from text.
    """

    model_dim: int
    input_modality: str
    speech_encoder_frontend: TransformerFrontend
    speech_encoder: TransformerEncoder
    text_encoder_frontend: Optional[TransformerFrontend]
    text_encoder: Optional[TransformerEncoder]
    text_decoder_frontend: Optional[TransformerFrontend]
    text_decoder: Optional[TransformerDecoder]
    final_proj: Optional[Projection]
    t2u_model: Union["UnitYT2UModel", "UnitYNART2UModel", None]
    prosody_encoder_model: Optional[ECAPA_TDNN]
    max_target_seq_len: int
    pad_idx: Optional[int]

    def __init__(
        self,
        speech_encoder_frontend: TransformerFrontend,
        speech_encoder: TransformerEncoder,
        text_encoder_frontend: Optional[TransformerFrontend],
        text_encoder: Optional[TransformerEncoder],
        text_decoder_frontend: Optional[TransformerFrontend],
        text_decoder: Optional[TransformerDecoder],
        final_proj: Optional[Projection],
        t2u_model: Union["UnitYT2UModel", "UnitYNART2UModel", None],
        max_target_seq_len: int,
        pad_idx: Optional[int],
        prosody_encoder_model: Optional[ECAPA_TDNN] = None,
        input_modality: str = "speech",
    ) -> None:
        super().__init__()

        self.model_dim = speech_encoder.model_dim
        self.max_target_seq_len = max_target_seq_len
        self.pad_idx = pad_idx

        self.input_modality = input_modality

        self.speech_encoder_frontend = speech_encoder_frontend
        self.speech_encoder = speech_encoder

        if text_encoder is not None:
            if text_encoder_frontend is None:
                raise ValueError(
                    "Both `text_encoder` and `text_encoder_frontend` must be specified, but `text_encoder_frontend` is `None`."
                )

            self.text_encoder_frontend = text_encoder_frontend
            self.text_encoder = text_encoder
        else:
            if text_encoder_frontend is not None:
                raise ValueError(
                    "Both `text_encoder` and `text_encoder_frontend` must be specified, but `text_encoder` is `None`."
                )

            self.register_module("text_encoder_frontend", None)
            self.register_module("text_encoder", None)

        if text_decoder is not None:
            if text_decoder_frontend is None:
                raise ValueError(
                    "Both `text_decoder` and `text_decoder_frontend` must be specified, but `text_decoder_frontend` is `None`."
                )

            self.text_decoder_frontend = text_decoder_frontend
            self.text_decoder = text_decoder
            self.final_proj = final_proj
        else:
            if text_decoder_frontend is not None:
                raise ValueError(
                    "Both `text_encoder` and `text_encoder_frontend` must be specified, but `text_decoder` is `None`."
                )

            self.register_module("text_decoder_frontend", None)
            self.register_module("text_decoder", None)
            self.register_module("final_proj", None)

        if t2u_model is not None:
            self.t2u_model = t2u_model
        else:
            self.register_module("t2u_model", None)

        if prosody_encoder_model is not None:
            self.prosody_encoder_model = prosody_encoder_model
        else:
            self.register_module("prosody_encoder_model", None)

    def encode(
        self, seqs: Tensor, seqs_layout: BatchLayout
    ) -> Tuple[Tensor, BatchLayout]:
        if self.input_modality == "speech":
            return self.encode_speech(seqs, seqs_layout)

        if self.input_modality == "text":
            return self.encode_text(seqs, seqs_layout)

        raise RuntimeError(
            f"`input_modality` must be 'speech' or 'text', but is '{self.input_modality}' instead."
        )

    def encode_speech(
        self, seqs: Tensor, seqs_layout: BatchLayout
    ) -> Tuple[Tensor, BatchLayout]:
        seqs, seqs_layout = self.speech_encoder_frontend(seqs, seqs_layout)

        return self.speech_encoder(seqs, seqs_layout)  # type: ignore[no-any-return]

    def encode_text(
        self, seqs: Tensor, seqs_layout: BatchLayout
    ) -> Tuple[Tensor, BatchLayout]:
        if self.text_encoder is None:
            raise ValueError(
                "`encode_text()` requires a text encoder, but the current UnitY model does not have one."
            )

        assert self.text_encoder_frontend is not None

        seqs, seqs_layout = self.text_encoder_frontend(seqs, seqs_layout)

        return self.text_encoder(seqs, seqs_layout)  # type: ignore[no-any-return]

    def decode(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        encoder_output: Tensor,
        encoder_output_layout: BatchLayout,
        *,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, BatchLayout]:
        if self.text_decoder is None:
            raise ValueError(
                "`decode()` requires a text decoder, but the current UnitY model does not have one."
            )

        assert self.text_decoder_frontend is not None

        seqs, seqs_layout = self.text_decoder_frontend(
            seqs, seqs_layout, state_bag=state_bag
        )

        return self.text_decoder(  # type: ignore[no-any-return]
            seqs,
            seqs_layout,
            encoder_output,
            encoder_output_layout,
            state_bag=state_bag,
        )

    def project(
        self, decoder_output: Tensor, decoder_output_layout: BatchLayout
    ) -> SequenceModelOutput:
        if self.final_proj is None:
            raise ValueError(
                "`project()` requires a final_proj layer, but the current UnitY model does not have one."
            )

        logits = self.final_proj(decoder_output)

        return SequenceModelOutput(logits, self.pad_idx)


@final
class UnitYX2TModel(Module):
    model_dim: int
    encoder_frontend: TransformerFrontend
    encoder: TransformerEncoder
    decoder_frontend: TransformerFrontend
    decoder: TransformerDecoder
    final_proj: Projection
    max_target_seq_len: int
    pad_idx: Optional[int]

    def __init__(
        self,
        encoder_frontend: TransformerFrontend,
        encoder: TransformerEncoder,
        decoder_frontend: TransformerFrontend,
        decoder: TransformerDecoder,
        final_proj: Projection,
        max_target_seq_len: int,
        pad_idx: Optional[int],
    ) -> None:
        super().__init__()

        self.model_dim = encoder.model_dim
        self.max_target_seq_len = max_target_seq_len
        self.pad_idx = pad_idx

        self.encoder_frontend = encoder_frontend
        self.encoder = encoder
        self.decoder_frontend = decoder_frontend
        self.decoder = decoder
        self.final_proj = final_proj

    def encode(
        self, seqs: Tensor, seqs_layout: BatchLayout
    ) -> Tuple[Tensor, BatchLayout]:
        seqs, seqs_layout = self.encoder_frontend(seqs, seqs_layout)
        return self.encoder(seqs, seqs_layout)  # type: ignore[no-any-return]

    def decode(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        encoder_output: Tensor,
        encoder_output_layout: BatchLayout,
        *,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, BatchLayout]:
        seqs, seqs_layout = self.decoder_frontend(
            seqs, seqs_layout, state_bag=state_bag
        )

        return self.decoder(  # type: ignore[no-any-return]
            seqs,
            seqs_layout,
            encoder_output,
            encoder_output_layout,
            state_bag=state_bag,
        )

    def project(
        self, decoder_output: Tensor, decoder_output_layout: BatchLayout
    ) -> SequenceModelOutput:
        logits = self.final_proj(decoder_output)

        return SequenceModelOutput(logits, self.pad_idx)


@final
class UnitYT2UModel(Module):
    """Represents a UnitY T2U model as described in
    :cite:t`https://doi.org/10.48550/arxiv.2212.08055`."""

    model_dim: int
    encoder: Optional[TransformerEncoder]
    decoder_frontend: TransformerFrontend
    decoder: TransformerDecoder
    final_proj: Projection
    max_target_seq_len: int
    pad_idx: Optional[int]

    def __init__(
        self,
        encoder: Optional[TransformerEncoder],
        decoder_frontend: TransformerFrontend,
        decoder: TransformerDecoder,
        final_proj: Projection,
        max_target_seq_len: int,
        pad_idx: Optional[int],
    ) -> None:
        super().__init__()

        self.model_dim = decoder.model_dim
        self.max_target_seq_len = max_target_seq_len
        self.pad_idx = pad_idx

        if encoder is not None:
            self.encoder = encoder
        else:
            self.register_module("encoder", None)

        self.decoder_frontend = decoder_frontend
        self.decoder = decoder

        self.final_proj = final_proj

    def encode(
        self, seqs: Tensor, seqs_layout: BatchLayout
    ) -> Tuple[Tensor, BatchLayout]:
        if self.encoder is None:
            return seqs, seqs_layout

        return self.encoder(seqs, seqs_layout)  # type: ignore[no-any-return]

    def decode(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        encoder_output: Tensor,
        encoder_output_layout: BatchLayout,
        *,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, BatchLayout]:
        seqs, seqs_layout = self.decoder_frontend(
            seqs, seqs_layout, state_bag=state_bag
        )

        return self.decoder(  # type: ignore[no-any-return]
            seqs,
            seqs_layout,
            encoder_output,
            encoder_output_layout,
            state_bag=state_bag,
        )

    def project(
        self, decoder_output: Tensor, decoder_output_layout: BatchLayout
    ) -> SequenceModelOutput:
        logits = self.final_proj(decoder_output)

        return SequenceModelOutput(logits, self.pad_idx)


@final
class UnitYNART2UModel(Module):
    """Represents a non-autoregressive UnitY T2U model."""

    model_dim: int
    encoder: Optional[TransformerEncoder]
    decoder_frontend: NARDecoderFrontend
    decoder: FeedForwardTransformer
    final_proj: Projection
    pad_idx: Optional[int]
    prosody_proj: Optional[Projection]

    def __init__(
        self,
        encoder: Optional[TransformerEncoder],
        decoder_frontend: NARDecoderFrontend,
        decoder: FeedForwardTransformer,
        final_proj: Projection,
        pad_idx: Optional[int],
        prosody_proj: Optional[Projection] = None,
    ) -> None:
        super().__init__()

        self.model_dim = decoder.model_dim

        if encoder is not None:
            if encoder.model_dim != self.model_dim:
                raise ValueError(
                    f"`model_dim` of `encoder` and `model_dim` of `decoder` must be equal, but are {encoder.model_dim} and {self.model_dim} instead."
                )

            self.encoder = encoder
        else:
            self.register_module("encoder", None)

        if decoder_frontend.model_dim != self.model_dim:
            raise ValueError(
                f"`model_dim` of `decoder_frontend` and `model_dim` of `decoder` must be equal, but are {decoder_frontend.model_dim} and {self.model_dim} instead."
            )

        self.decoder_frontend = decoder_frontend
        self.decoder = decoder

        self.final_proj = final_proj
        self.pad_idx = pad_idx
        self.prosody_proj = prosody_proj

    def forward(
        self,
        text_decoder_output: Tensor,
        text_decoder_layout: BatchLayout,
        text_seqs: Optional[Tensor],
        duration_factor: float = 1.0,
        film_cond_emb: Optional[Tensor] = None,
    ) -> Tuple[SequenceModelOutput, BatchLayout, Tensor]:
        encoder_output, encoder_layout = self.encode(
            text_decoder_output, text_decoder_layout
        )

        if self.prosody_proj is not None and film_cond_emb is not None:
            encoder_output = encoder_output + self.prosody_proj(film_cond_emb)

        decoder_output, decoder_layout, durations = self.decode(
            encoder_output,
            encoder_layout,
            text_seqs,
            duration_factor,
            film_cond_emb,
        )

        return self.project(decoder_output), decoder_layout, durations

    def encode(
        self,
        text_decoder_output: Tensor,
        text_decoder_layout: BatchLayout,
    ) -> Tuple[Tensor, BatchLayout]:
        if self.encoder is None:
            return text_decoder_output, text_decoder_layout

        return self.encoder(text_decoder_output, text_decoder_layout)  # type: ignore[no-any-return]

    def decode(
        self,
        encoder_output: Tensor,
        encoder_layout: BatchLayout,
        text_seqs: Optional[Tensor],
        duration_factor: float = 1.0,
        film_cond_emb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, BatchLayout, Tensor]:
        seqs, layout, durations = self.decoder_frontend(
            encoder_output,
            encoder_layout,
            text_seqs,
            duration_factor,
            film_cond_emb,
        )

        seqs, layout = self.decoder(
            seqs, layout, film_cond_emb=film_cond_emb
        )

        return seqs, layout, durations  # type: ignore[no-any-return]

    def project(self, decoder_output: Tensor) -> SequenceModelOutput:
        logits = self.final_proj(decoder_output)

        return SequenceModelOutput(logits, self.pad_idx)


@dataclass
class UnitYOutput:
    """Holds the output of a UnitY model."""

    s2t_output: SequenceModelOutput
    """The S2T output of the multitask model."""

    mt_output: SequenceModelOutput
    """The MT output of the multitask model."""

    t2u_output: SequenceModelOutput
    """The output of the T2U model."""

    def compute_loss(
        self, targets: Tensor, ignore_prefix_size: int = 0, label_smoothing: float = 0.0
    ) -> None:
        pass
