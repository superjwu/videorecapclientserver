import math
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange, repeat

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers.generation.logits_process import (
    LogitsProcessorList, TopKLogitsWarper, TopPLogitsWarper,
    TemperatureLogitsWarper, TypicalLogitsWarper, LogitNormalization
)
from collections import OrderedDict

from transformers import GPT2LMHeadModel
from .gpt2_gated import GPT2LMHeadModel as GatedGPT2LMHeadModel
from .gpt2_gated import augment_gpt2_config
from .model_utils import rsetattr, remap_keys
from .openai_model import QuickGELU, Transformer
from .openai_clip import load as load_openai_clip
from .timesformer import SpaceTimeTransformer

from .mappers import get_mapper

from huggingface_hub import PyTorchModelHubMixin

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class VideoRecap(nn.Module):
    def __init__(self, args, use_vision_model_forced=False, eval_only=False):
        super().__init__()

        if args.video_feature_type == 'pixel' or use_vision_model_forced:
            self.vision_model_type = args.vision_model_type
            if args.vision_model_type == 'clip_b16':
                clip_model, _ = load_openai_clip('ViT-B/16', 'cpu')
                self.vision_model = clip_model.visual
            else:
                vision_model = SpaceTimeTransformer(
                    num_frames=args.num_video_feat,
                    time_init='zeros',
                    attention_style='frozen-in-time',
                    ln_pre=True,
                    act_layer=QuickGELU,
                    is_tanh_gating=False,
                )

                if not eval_only:
                    clip_model, _ = load_openai_clip('ViT-B/16', 'cpu')
                    print("=> Loading CLIP (ViT-B/16) weights")
                    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=12)
                    res = vision_model.load_state_dict(remapped_state_dict, strict=False)

                vision_model.head = nn.Identity()
                vision_model.pre_logits = nn.Identity()
                vision_model.fc = nn.Identity()

                if not eval_only:
                    # Freeze visual encoder
                    for n, p in vision_model.named_parameters():
                        p.requires_grad = False
                    # Load contrastive VLP pretrained weights
                    print('Loading Video encoder from', args.video_encoder_ckpt)
                    checkpoint = torch.load(args.video_encoder_ckpt, map_location='cpu')
                    state_dict = OrderedDict()
                    for k, v in checkpoint['state_dict'].items():
                        if 'visual' in k:
                            state_dict[k.replace('module.visual.', '')] = v
                    vision_model.load_state_dict(state_dict, strict=True)

                self.vision_model = vision_model

        if args.video_mapper_type is not None:
            self.video_queries = nn.Parameter(torch.empty(args.num_video_queries, args.query_width))
            nn.init.normal_(self.video_queries, std=args.query_width ** -0.5)

            self.video_mapper = get_mapper(
                args.video_mapper_type,
                args.num_video_queries,
                args.query_width,
                args.video_feature_width,
                finetune_mapper=args.finetune_mapper
            )

        if args.text_mapper_type is not None:
            self.text_queries = nn.Parameter(torch.empty(args.num_text_queries, args.query_width))
            nn.init.normal_(self.text_queries, std=args.query_width ** -0.5)

            if args.share_mapper:
                print("Sharing video and text mapper")
                self.text_mapper = self.video_mapper
            else:
                self.text_mapper = get_mapper(
                    args.text_mapper_type,
                    args.num_text_queries,
                    args.query_width,
                    args.text_width,
                    finetune_mapper=args.finetune_mapper
                )

        # The following lines related to self.text_decoder have been removed
        # as per your request.

    def map_features(self, samples):
        queries = []
        if hasattr(self, "video_queries") and "video_features" in samples:
            batch_size = samples["video_features"].shape[0]
            video_queries = repeat(self.video_queries, 'n d -> b n d', b=batch_size)
            video_features = samples["video_features"].to(video_queries.device, dtype=video_queries.dtype)

            if "video_mask" in samples:
                attention_mask = samples["video_mask"].to(video_queries.device)
                video_queries = self.video_mapper(video_queries, video_features, attention_mask)
            else:
                video_queries = self.video_mapper(video_queries, video_features)

            queries.append(video_queries)

        if hasattr(self, "text_queries") and "text_features" in samples:
            batch_size = samples["text_features"].shape[0]
            text_queries = repeat(self.text_queries, 'n d -> b n d', b=batch_size)
            # Removed usage of self.text_decoder
            # text_features = self.text_decoder.transformer.wte(samples["text_features"].to(text_queries.device))
            # Replaced with appropriate embedding layer or processing
            text_features = samples["text_features"].to(text_queries.device)

            position_ids = torch.arange(0, text_features.shape[1], dtype=torch.long)
            pe = PositionalEncoding(text_features.shape[-1]).to(text_queries.device)
            text_features = text_features + pe(position_ids)

            attention_mask = samples["text_mask"].to(text_queries.device)
            text_queries = self.text_mapper(text_queries, text_features, attention_mask)
            queries.append(text_queries)

        queries = torch.cat(queries, dim=1)
        return queries

    def forward(self, samples, use_checkpoint=False):
        if "video_features" in samples and len(samples["video_features"].shape) == 5:
            if self.vision_model_type == 'clip_b16':
                image = rearrange(samples["video_features"], 'b t c h w-> (b t) c h w')
                image = self.vision_model(image, cls_at_last=False)
                samples["video_features"] = image.reshape(samples["video_features"].shape[0], -1, image.shape[-1])
            else:
                image = samples["video_features"].permute(0, 2, 1, 3, 4).contiguous()  # BCTHW -> BTCHW
                samples["video_features"] = self.vision_model.forward_features(
                    image,
                    use_checkpoint=use_checkpoint,
                    cls_at_last=False
                )  # NLD

        queries = self.map_features(samples)

        # Since self.text_decoder has been removed, the following code that uses it
        # has been commented out or needs to be replaced

        # if use_checkpoint:
        #     self.text_decoder.gradient_checkpointing_enable()
        # else:
        #     self.text_decoder.gradient_checkpointing_disable()

        # text = samples["caption_tokens"]
        # text, labels = text[:, :-1], text[:, 1:]

        # output_decoder = self.text_decoder(text.contiguous(), encoder_hidden_states=queries)
        # text_tokens_logits = output_decoder.logits

        # text_tokens_logits = rearrange(text_tokens_logits, 'b n c -> b c n').to(labels.device)
        # loss = F.cross_entropy(text_tokens_logits, labels, ignore_index=0, reduction='none')
        # loss = loss.mean()

        # For now, returning None or some placeholder loss
        loss = None

        return loss

    # The generate method and _get_logits_warper method have been removed or need to be adjusted
    # because they rely on self.text_decoder

    # def generate(self, ...):
    #     # Method implementation

    # def _get_logits_warper(self, ...):
    #     # Method implementation