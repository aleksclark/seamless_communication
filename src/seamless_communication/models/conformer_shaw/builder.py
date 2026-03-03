# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from dataclasses import asdict, dataclass, field
from typing import Final, Optional

from fairseq2.models.conformer import ConformerConvolution

from fairseq2.models.w2vbert import W2VBertConfig
from fairseq2.models.wav2vec2 import (
    Wav2Vec2Config,
    Wav2Vec2EncoderConfig,
    Wav2Vec2EncoderFactory,
    Wav2Vec2Factory,
)
from fairseq2.models.wav2vec2.model import Wav2Vec2Model
from fairseq2.models.transformer import SDPA, ShawRelativePositionSDPA, create_default_sdpa
from fairseq2.data_type import DataType
from fairseq2.device import Device

CONFORMER_SHAW_FAMILY: Final = "conformer_shaw"


@dataclass
class ShawRelativePositionSDPAConfig:
    """Holds the configuration of the :class:ShawRelativePositionSDPA module."""

    max_left_rel_pos: int = 64
    """The left clipping value for relative positions."""

    max_right_rel_pos: Optional[int] = 8
    """The right clipping value for relative positions."""

    use_rel_pos_values: bool = False
    """If True, also uses relative position values to compute relative attention."""


@dataclass
class ConformerShawEncoderConfig(Wav2Vec2EncoderConfig):
    """Holds the configuration of a conformer shaw encoder."""

    shaw_rel_pos_sdpa_config: Optional[ShawRelativePositionSDPAConfig] = None
    """The parameters for ShawRelativePositionSDPA."""


@dataclass
class ConformerShawConfig(Wav2Vec2Config):
    """Holds the configuration of a conformer shaw model."""

    encoder_config: ConformerShawEncoderConfig = field(
        default_factory=ConformerShawEncoderConfig
    )


conformer_shaw_archs: dict = {}

def conformer_shaw_arch(name):
    def decorator(fn):
        conformer_shaw_archs[name] = fn
        return fn
    return decorator

def _conformer_shaw_600m_encoder() -> ConformerShawEncoderConfig:
    w2vbert_config = W2VBertConfig()
    w2v2_encoder_config = w2vbert_config.w2v2_config.encoder_config
    sdpa_config = ShawRelativePositionSDPAConfig(
        max_left_rel_pos=64,
        max_right_rel_pos=8,
        use_rel_pos_values=False,
    )
    conformer_shaw_encoder_config = ConformerShawEncoderConfig(
        **asdict(w2v2_encoder_config),
        shaw_rel_pos_sdpa_config=sdpa_config,
    )
    conformer_shaw_encoder_config.pos_encoder_type = "shaw_relative"
    return conformer_shaw_encoder_config


@conformer_shaw_arch("conformer_shaw_600m")
def _conformer_shaw_600m() -> ConformerShawConfig:
    encoder_config = _conformer_shaw_600m_encoder()

    return ConformerShawConfig(
        encoder_config,
        final_dim=768,
        final_proj_bias=True,
        temporal_mask_span_len=10,
        max_temporal_mask_prob=0.65,
        min_num_temporal_mask_spans=2,
        spatial_mask_span_len=10,
        max_spatial_mask_prob=0.0,
        min_num_spatial_mask_spans=2,
        quantized_dim=768,
        num_codebooks=2,
        num_codebook_entries=320,
        codebook_sampling_temperature=(2.0, 0.1, 0.999995),
        num_distractors=100,
        logit_temp=0.1,
        diversity_loss_weight=0.2,
    )


class ConformerShawEncoderBuilder(Wav2Vec2EncoderFactory):
    """
    Builds modules of a `ConformerShawEncoderBuilder`.

    This is a Conformer architecture with these differences:
    - ShawRelativePositionSDPA as the SDPA.
    - ConformerConvolution with causal depthwise convolution
    and norm_type "layer_norm".
    """

    config: ConformerShawEncoderConfig
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: ConformerShawEncoderConfig,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        super().__init__(config)

        self.config = config

        assert self.config.use_conformer, "This architecture only supports a Conformer."
        assert (
            self.config.pos_encoder_type == "shaw_relative"
        ), "This architecture only supports ShawRelativePositionSDPA."

        self._device, self._dtype = device, dtype

    def create_rel_pos_encoding(self) -> ShawRelativePositionSDPA:
        if self.config.shaw_rel_pos_sdpa_config is None:
            raise ValueError(
                "`shaw_rel_pos_sdpa_config` must be specified when `pos_encoder_type` is 'shaw_relative'."
            )

        sdpa = create_default_sdpa(attn_dropout_p=self.config.attn_dropout_p)

        sdpa_config = self.config.shaw_rel_pos_sdpa_config

        return ShawRelativePositionSDPA(
            self.config.model_dim,
            self.config.num_encoder_attn_heads,
            sdpa_config.max_left_rel_pos,
            max_right_rel_pos=sdpa_config.max_right_rel_pos,
            use_rel_pos_values=sdpa_config.use_rel_pos_values,
            inner_sdpa=sdpa,
            device=self._device,
            dtype=self._dtype,
        )

    def create_conformer_conv(self) -> ConformerConvolution:
        return ConformerConvolution(
            self.config.model_dim,
            self.config.depthwise_conv_kernel_size,
            causal_depthwise_conv=True,
            norm_type="layer_norm",
            device=self._device,
            dtype=self._dtype,
        )


def create_conformer_shaw_model(
    config: ConformerShawConfig,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> Wav2Vec2Model:
    """Create a conformer shaw model.

    :param config:
        The configuration.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    encoder_builder = ConformerShawEncoderBuilder(
        config.encoder_config
    )

    builder = Wav2Vec2Factory(
        config
    )
    builder._encoder_factory = encoder_builder

    return builder.create_model(device=device, dtype=dtype)
