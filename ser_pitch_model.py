import torch.nn as nn
import torch     
from transformers import HubertModel, Wav2Vec2PreTrainedModel, HubertPreTrainedModel, Wav2Vec2Config, HubertConfig, Wav2Vec2Model
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2FeatureEncoder, Wav2Vec2FeatureProjection
from transformers.modeling_outputs import CausalLMOutput, SequenceClassifierOutput
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

import math
from typing import Any, Dict, List, Optional, Union


_HIDDEN_STATES_START_POSITION = 2
pretrain_processor_wav2vec2 = 'facebook/wav2vec2-base-100h'
pretrain_audio_model_wav2vec2 = 'facebook/wav2vec2-base'

        
class AttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, tar):
        src = src.transpose(0, 1) # B, T, C -> T, B, C
        tar = tar.transpose(0, 1) # B, T, C -> T, B, C
        src2 = self.self_attn(tar, src, src, attn_mask=None,
                              key_padding_mask=None)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src.transpose(0, 1) # T, B, C -> B, T, C
        return src        

class Wav2Vec2CrossAttentionPitchForSER(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        
        # self.AcousticEncoder = AcousticEncoder()
        # self.PitchEncoder = PitchEncoder()
        
        # CNN for pitch encoder
        self.pitch_encoder = Wav2Vec2FeatureEncoder(config)
        self.pitch_projection = Wav2Vec2FeatureProjection(config)

        # attn -> concat -> linear -> prediction 
        self.attn_a_p = AttentionLayer(config.hidden_size, 16)
        self.attn_p_a = AttentionLayer(config.hidden_size, 16)
        self.self_attn = AttentionLayer((config.hidden_size * 2), 16)
        
        self.projector = nn.Linear((config.hidden_size * 2), config.classifier_proj_size)
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)
        
        self.post_init()
        
    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(self, audio_input, pitch_input, label=None, return_dict=None):
        # acoustic output
        acoustic = self.wav2vec2(audio_input.input_values, attention_mask=None)[0]
        # acoustic = self.AcousticEncoder(wav2vec_output) #batch x time x 768
        
        # pitch output
        pitch = self.pitch_encoder(pitch_input.input_values)
        pitch = pitch.transpose(1, 2)
        pitch, _ = self.pitch_projection(pitch)
        
        a_p_attn = self.attn_a_p(acoustic, pitch)
        p_a_attn = self.attn_p_a(pitch, acoustic)

        
        attention_output = torch.cat((a_p_attn, p_a_attn), -1)    
        attention_output = self.self_attn(attention_output, attention_output)       

        attention_output = self.projector(attention_output)
        attention_output = attention_output.mean(dim=1)
        logits = self.classifier(attention_output)
        
        loss = None
        if label is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), label.view(-1))

        # if not return_dict:
        #     output = (logits,) + acoustic[_HIDDEN_STATES_START_POSITION:]
        #     return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )