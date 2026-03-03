# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Union

from fairseq2.models.conformer import ConformerBlock, ConformerConvolution
from fairseq2.models.nllb import NllbConfig, NllbFactory
from fairseq2.models.wav2vec2 import Wav2Vec2EncoderConfig, Wav2Vec2EncoderFactory
from fairseq2.models.transformer import (
    FeedForwardNetwork,
    MultiheadAttention,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerNormOrder,
    create_default_sdpa,
)
from fairseq2.models.transformer.attention_bias import IdentityBias
from fairseq2.nn import TiedProjection
from fairseq2.data_type import DataType
from fairseq2.device import Device
from typing_extensions import override
from torch.nn import GELU, ReLU

from seamless_communication.models.generator.ecapa_tdnn_builder import (
    EcapaTDNNBuilder,
    EcapaTDNNConfig,
)
from seamless_communication.models.unity.adaptor_block import (
    UnitYConformerAdaptorLayer,
    UnitYEncoderAdaptor,
    UnitYTransformerAdaptorLayer,
)
from seamless_communication.models.unity.model import UNITY_FAMILY, UnitYModel
from seamless_communication.models.unity.t2u_builder import (
    UnitYNART2UBuilder,
    UnitYT2UBuilder,
    UnitYT2UConfig,
)
from seamless_communication.models.conformer_shaw import (
    ConformerShawEncoderBuilder,
    ConformerShawEncoderConfig,
)


@dataclass
class UnitYConfig:
    """Holds the configuration of a UnitY model as described in
    :cite:t`https://doi.org/10.48550/arxiv.2212.08055`"""

    model_dim: int
    """The dimensionality of the model."""

    w2v2_encoder_config: Wav2Vec2EncoderConfig
    """The configuration of the underlying wav2vec 2.0 encoder."""

    nllb_config: NllbConfig
    """The configuration of the underlying MT text encoder-decoder."""

    t2u_config: Optional[UnitYT2UConfig]
    """The configuration of the UnitY T2U sub-model."""

    prosody_encoder_config: Optional[EcapaTDNNConfig]
    """The configuration of the expressive prosody encoder."""

    use_text_encoder: bool
    """If ``True``, uses an aligned MT encoder for the MT task."""

    use_text_decoder: bool
    """If ``False``, skips loading a text decoder, to be used with a Monotonic decoder."""

    use_conformer_adaptor: bool
    """If ``True``, uses a Conformer-based adaptor block."""

    use_gelu: bool
    """If ``True``, uses GELU activation function in feed-forward networks of
    adaptor blocks and decoder layers."""

    num_adaptor_layers: int
    """The number of Transformer encoder layers in the adaptor block."""

    adaptor_kernel_size: int
    """The kernel size of 1D convolutions in the adaptor block."""

    adaptor_stride: int
    """The stride of 1D convolutions in the adaptor block."""

    adaptor_layer_norm: bool
    """If ``True``, applies Layer Normalization to outputs of the underlying
    encoder in the adaptor block."""

    adaptor_dropout_p: float
    """The dropout probability in Transformer layers of the adaptor block."""


def _nllb_dense_1b() -> NllbConfig:
    return NllbConfig()


def _nllb_dense_600m() -> NllbConfig:
    config = _nllb_dense_1b()
    config.num_encoder_layers = 12
    config.num_decoder_layers = 12
    config.ffn_inner_dim = 1024 * 4
    return config


def _base_unity_config() -> UnitYConfig:
    nllb_config = _nllb_dense_1b()
    nllb_config.vocab_size = 256102  # NLLB-100

    return UnitYConfig(
        model_dim=1024,
        w2v2_encoder_config=Wav2Vec2EncoderConfig(),
        nllb_config=nllb_config,
        t2u_config=None,
        prosody_encoder_config=None,
        use_text_encoder=True,
        use_text_decoder=True,
        use_conformer_adaptor=False,
        use_gelu=False,
        num_adaptor_layers=1,
        adaptor_kernel_size=8,
        adaptor_stride=8,
        adaptor_layer_norm=True,
        adaptor_dropout_p=0.1,
    )


UNITY_ARCHS: dict[str, type] = {}


def register_unity_arch(name: str):
    def decorator(fn):
        UNITY_ARCHS[name] = fn
        return fn
    return decorator


def get_unity_config(arch: str) -> UnitYConfig:
    if arch not in UNITY_ARCHS:
        raise ValueError(f"Unknown unity arch: {arch}")
    return UNITY_ARCHS[arch]()


@register_unity_arch("base")
def _base() -> UnitYConfig:
    return _base_unity_config()


@register_unity_arch("medium")
def _medium() -> UnitYConfig:
    nllb_config = _nllb_dense_600m()
    nllb_config.vocab_size = 256206  # NLLB-200

    return UnitYConfig(
        model_dim=1024,
        w2v2_encoder_config=Wav2Vec2EncoderConfig(),
        nllb_config=nllb_config,
        t2u_config=None,
        prosody_encoder_config=None,
        use_text_encoder=True,
        use_text_decoder=True,
        use_conformer_adaptor=False,
        use_gelu=False,
        num_adaptor_layers=1,
        adaptor_kernel_size=8,
        adaptor_stride=8,
        adaptor_layer_norm=True,
        adaptor_dropout_p=0.1,
    )


@register_unity_arch("base_v2")
def _base_v2() -> UnitYConfig:
    nllb_config = _nllb_dense_1b()
    nllb_config.vocab_size = 256102  # NLLB-100
    nllb_config.max_seq_len = 4096

    return UnitYConfig(
        model_dim=1024,
        w2v2_encoder_config=Wav2Vec2EncoderConfig(),
        nllb_config=nllb_config,
        t2u_config=None,
        prosody_encoder_config=None,
        use_text_encoder=True,
        use_text_decoder=True,
        use_conformer_adaptor=False,
        use_gelu=False,
        num_adaptor_layers=1,
        adaptor_kernel_size=8,
        adaptor_stride=8,
        adaptor_layer_norm=True,
        adaptor_dropout_p=0.1,
    )


@register_unity_arch("expressivity_v2")
def _expressivity_v2() -> UnitYConfig:
    nllb_config = _nllb_dense_1b()
    nllb_config.vocab_size = 256102  # NLLB-100
    nllb_config.max_seq_len = 10000

    return UnitYConfig(
        model_dim=1024,
        w2v2_encoder_config=Wav2Vec2EncoderConfig(),
        nllb_config=nllb_config,
        t2u_config=None,
        prosody_encoder_config=EcapaTDNNConfig(),
        use_text_encoder=False,
        use_text_decoder=True,
        use_conformer_adaptor=False,
        use_gelu=True,
        num_adaptor_layers=1,
        adaptor_kernel_size=8,
        adaptor_stride=8,
        adaptor_layer_norm=True,
        adaptor_dropout_p=0.1,
    )


class UnitYBuilder:
    """Builds modules of a UnitY model.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    config: UnitYConfig
    w2v2_encoder_factory: Wav2Vec2EncoderFactory
    nllb_factory: NllbFactory
    t2u_builder: Union[UnitYT2UBuilder, UnitYNART2UBuilder, None]
    prosody_encoder_builder: Optional[EcapaTDNNBuilder]
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: UnitYConfig,
        w2v2_encoder_factory: Wav2Vec2EncoderFactory,
        nllb_factory: NllbFactory,
        t2u_builder: Union[UnitYT2UBuilder, UnitYNART2UBuilder, None],
        prosody_encoder_builder: Optional[EcapaTDNNBuilder],
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        if config.w2v2_encoder_config.model_dim != config.model_dim:
            raise ValueError(
                f"`config.model_dim` and `config.w2v2_encoder_config.model_dim` must be equal, but are {config.model_dim} and {config.w2v2_encoder_config.model_dim} instead."
            )

        if config.nllb_config.model_dim != config.model_dim:
            raise ValueError(
                f"`config.model_dim` and `config.nllb_config.model_dim` must be equal, but are {config.model_dim} and {config.nllb_config.model_dim} instead."
            )

        if config.t2u_config is not None and config.t2u_config.model_dim != config.model_dim:
            raise ValueError(
                f"`config.model_dim` and `config.t2u_config.model_dim` must be equal, but are {config.model_dim} and {config.t2u_config.model_dim} instead."
            )

        self.config = config

        self.w2v2_encoder_factory = w2v2_encoder_factory
        self.nllb_factory = nllb_factory
        self.t2u_builder = t2u_builder
        self.prosody_encoder_builder = prosody_encoder_builder

        self.device, self.dtype = device, dtype

    def build_model(self) -> UnitYModel:
        """Build a model."""
        speech_encoder_frontend = self.w2v2_encoder_factory.create_encoder_frontend()
        speech_encoder = self.build_speech_encoder()

        if self.config.use_text_encoder:
            text_embed = self.nllb_factory.create_embedding()
            text_encoder_frontend = self.nllb_factory.create_frontend(text_embed)
            text_encoder = self.nllb_factory.create_encoder()
        else:
            text_embed = None
            text_encoder_frontend = None
            text_encoder = None

        if self.config.use_text_decoder:
            if text_embed is None:
                text_embed = self.nllb_factory.create_embedding()

            if text_encoder_frontend is not None:
                text_decoder_frontend = text_encoder_frontend
            else:
                text_decoder_frontend = self.nllb_factory.create_frontend(text_embed)

            text_decoder = self.nllb_factory.create_decoder()
            final_proj = TiedProjection(text_embed.weight, bias=None)
        else:
            text_decoder_frontend = None
            text_decoder = None
            final_proj = None

        if self.t2u_builder is None:
            t2u_model = None
        else:
            t2u_model = self.t2u_builder.build_model()

        if self.prosody_encoder_builder is None:
            prosody_encoder_model = None
        else:
            prosody_encoder_model = self.prosody_encoder_builder.build_model()

        return UnitYModel(
            speech_encoder_frontend,
            speech_encoder,
            text_encoder_frontend,
            text_encoder,
            text_decoder_frontend,
            text_decoder,
            final_proj,
            t2u_model,
            self.config.nllb_config.max_seq_len or 0,
            self.config.nllb_config.pad_idx,
            prosody_encoder_model,
        )

    def build_speech_encoder(self) -> TransformerEncoder:
        """Build a speech Transformer encoder."""
        w2v2_encoder = self.w2v2_encoder_factory.create_encoder()

        if not self.config.use_conformer_adaptor:
            build_adaptor_layer = self.build_adaptor_layer
        else:
            build_adaptor_layer = self.build_conformer_adaptor_layer

        num_layers = self.config.num_adaptor_layers

        layers = [build_adaptor_layer(i) for i in range(num_layers)]

        return UnitYEncoderAdaptor(
            w2v2_encoder,
            layers,
            inner_layer_norm=self.config.adaptor_layer_norm,
            device=self.device,
            dtype=self.dtype,
        )

    def build_adaptor_layer(self, idx: int) -> TransformerEncoderLayer:
        """Build a Transformer-based encoder adaptor layer."""
        self_attn = self.build_adaptor_attention(
            self.config.w2v2_encoder_config.num_encoder_attn_heads
        )

        ffn = StandardFeedForwardNetwork(
            self.config.model_dim,
            self.config.w2v2_encoder_config.ffn_inner_dim,
            inner_activation=GELU() if self.config.use_gelu else ReLU(),
            bias=True,
            device=self.device,
            dtype=self.dtype,
        )

        return UnitYTransformerAdaptorLayer(
            self_attn,
            ffn,
            self.config.adaptor_kernel_size,
            self.config.adaptor_stride,
            dropout_p=self.config.adaptor_dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_conformer_adaptor_layer(self, idx: int) -> TransformerEncoderLayer:
        """Build a Conformer-based encoder adaptor layer."""
        ffn1 = self.w2v2_encoder_factory.create_ffn(use_swish=True)

        self_attn = self.build_adaptor_attention(
            self.config.w2v2_encoder_config.num_encoder_attn_heads
        )

        conv = ConformerConvolution(
            self.config.w2v2_encoder_config.model_dim,
            self.config.w2v2_encoder_config.depthwise_conv_kernel_size,
            device=self.device,
            dtype=self.dtype,
        )

        ffn2 = self.w2v2_encoder_factory.create_ffn(use_swish=True)

        block = ConformerBlock(
            ffn1,
            self_attn,
            conv,
            ffn2,
            dropout_p=self.config.adaptor_dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

        layer_norm = idx == 0

        return UnitYConformerAdaptorLayer(
            block,
            self.config.adaptor_kernel_size,
            self.config.adaptor_stride,
            layer_norm=layer_norm,
            device=self.device,
            dtype=self.dtype,
        )

    def build_adaptor_attention(self, num_heads: int) -> MultiheadAttention:
        """Build a Transformer multi-head attention layer in adaptor block."""
        attn_bias = IdentityBias()
        sdpa = create_default_sdpa(attn_bias)

        return StandardMultiheadAttention(
            self.config.model_dim,
            num_heads,
            sdpa=sdpa,
            device=self.device,
            dtype=self.dtype,
        )


class NllbWithGELUFactory(NllbFactory):
    @override
    def create_ffn(self) -> FeedForwardNetwork:
        return StandardFeedForwardNetwork(
            self._config.model_dim,
            self._config.ffn_inner_dim,
            bias=True,
            inner_activation=GELU(),
            norm_order=TransformerNormOrder.PRE,
            device=self._device if hasattr(self, '_device') else None,
            dtype=self._dtype if hasattr(self, '_dtype') else None,
        )


def create_unity_model(
    config: UnitYConfig,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> UnitYModel:
    """Create a UnitY model.

    :param config:
        The configuration to use.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    if isinstance(config.w2v2_encoder_config, ConformerShawEncoderConfig):
        w2v2_encoder_factory: Wav2Vec2EncoderFactory = ConformerShawEncoderBuilder(
            config.w2v2_encoder_config, device=device, dtype=dtype
        )
    else:
        w2v2_encoder_factory = Wav2Vec2EncoderFactory(
            config.w2v2_encoder_config,
        )

    t2u_builder: Union[UnitYT2UBuilder, UnitYNART2UBuilder, None]

    if config.t2u_config is None:
        t2u_builder = None
    elif config.t2u_config.nar_decoder_config is None:
        t2u_builder = UnitYT2UBuilder(config.t2u_config, device=device, dtype=dtype)
    else:
        t2u_builder = UnitYNART2UBuilder(config.t2u_config, device=device, dtype=dtype)

    if config.prosody_encoder_config is None:
        prosody_encoder_builder = None
    else:
        prosody_encoder_builder = EcapaTDNNBuilder(
            config.prosody_encoder_config, device=device, dtype=dtype
        )

    if config.use_gelu:
        nllb_factory: NllbFactory = NllbWithGELUFactory(config.nllb_config)
    else:
        nllb_factory = NllbFactory(config.nllb_config)

    unity_builder = UnitYBuilder(
        config,
        w2v2_encoder_factory,
        nllb_factory,
        t2u_builder,
        prosody_encoder_builder,
        device=device,
        dtype=dtype,
    )

    return unity_builder.build_model()
