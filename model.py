from torch.nn import CrossEntropyLoss
from torchcrf import CRF
import torch
import torch.nn as nn
from transformers import AutoModel, ViTModel
from utils import *
import copy
import torch.nn.functional as F
import math
from bert_model import *

class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x
    
class features_attention(nn.Module):
    def __init__(self, config, feature_attention=True):
        super(features_attention, self).__init__()
        self.config = config
        self.feature_attention = feature_attention
        if feature_attention:
            self.query = nn.Linear(config.latent, config.latent)
            self.key = nn.Linear(config.latent, config.latent)
            self.value = nn.Linear(config.latent, config.latent)
        else:
            self.query = nn.Linear(config.latent, config.latent)
            self.key = MLP(config.latent, config.latent, config.dropout)
            self.value = MLP(config.latent, config.latent, config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.vae = VAE(config.bert_hid_size, config.latent)

    def forward(self, main_feature, feature, is_img=False):
        """
            main_feature: [batch_size, N1, hidden_dim]
            feature: [batch_size, N2, hidden_dim]
        """
        main_feature = self.vae.encoder(main_feature)
        feature = self.vae.encoder(feature)
        if self.feature_attention:
            query_layer = self.query(main_feature.float()) # [batch_size, N2, hidden_dim]
            key_layer = self.key(feature.float()) # [batch_size, N1, hidden_dim]
            value_layer = self.value(feature.float()) # [batch_size, N1, hidden_dim]
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [bs, N2, N1]
            attention_probs = nn.Softmax(dim=-1)(attention_scores) # [bs, N2]
            attention_probs = self.dropout(attention_probs) # [bs, N2]
            attention_result = torch.matmul(attention_probs, value_layer) # [bs, N2, hidden_dim]
            attention_result = self.vae.decoder(attention_result)
            return attention_result
        else:
            new_main_feature = torch.zeros_like(main_feature).to(self.config.device)
            new_main_feature[:, :, :] = main_feature[:, :, :]
            new_main_feature_head = torch.zeros_like(main_feature[:, 0, :].unsqueeze(1).float()).to(self.config.device)
            new_main_feature_head[:, 0, :] = main_feature[:, 0, :]
            query_layer = self.query(new_main_feature_head.float()) # [batch_size, 1, hidden_dim]
            key_layer = self.key(feature.float()) # [batch_size, n, hidden_dim]
            value_layer = self.value(feature.float()) # [batch_size, n, hidden_dim]
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [bs, 1, n]
            attention_probs = nn.Softmax(dim=-1)(attention_scores) # [bs, 1]
            attention_probs = self.dropout(attention_probs) # [bs, 1]
            attention_result = torch.matmul(attention_probs, value_layer) # [bs, 1, hidden_dim]
            new_main_feature[:, 0, :] = attention_result.squeeze(1)
            new_main_feature = self.vae.decoder(new_main_feature)
            return new_main_feature

class FeedForwardLayer(nn.Module):

    def __init__(self, hidden_size, dropout=0.5):
        super(FeedForwardLayer, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        self.intermediate_layer = IntermediateLayer(self.hidden_size)
        self.intermediate_output_layer = IntermediateOutputLayer(self.hidden_size, self.dropout_rate)

    def forward(self, hidden_states):
        hidden_states1 = self.intermediate_layer(hidden_states)
        hidden_states = self.intermediate_output_layer(hidden_states1, hidden_states)
        return hidden_states

class IntermediateLayer(nn.Module):

    def __init__(self, hidden_size):
        super(IntermediateLayer, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = self.hidden_size // 2
        self.dense = nn.Linear(self.hidden_size, self.intermediate_size)
        self.activate_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.activate_fn(self.dense(hidden_states))
        return hidden_states


class IntermediateOutputLayer(nn.Module):

    def __init__(self, hidden_size, dropout=0.5):
        super(IntermediateOutputLayer, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = self.hidden_size // 2
        self.dropout_rate = dropout
        self.dense = nn.Linear(self.intermediate_size, self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, hidden_states, inputs):
        hidden_states = self.dropout(self.dense(hidden_states))
        hidden_states = self.layer_norm(hidden_states + inputs)
        return hidden_states

def clone_module(module: nn.Module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

class CrossMultiHeadAttention(nn.Module):

    def __init__(self, hidden_size, num_heads=16, dropout_rate=0.5):
        super(CrossMultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        assert self.hidden_size % self.num_heads == 0
        self.head_size = self.hidden_size // self.num_heads
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)
        self.linears = clone_module(nn.Linear(self.hidden_size, self.hidden_size), 3)

    def forward(self, input_q, input_k, input_v):
        bs, sl1, hs = input_q.shape
        _, sl2, hs = input_k.shape
        assert hs == self.hidden_size

        q, k, v = [
            layer(x).view(bs, -1, self.num_heads, self.head_size).transpose(1, 2)
            for layer, x in zip(self.linears, (input_q, input_k, input_v))
        ]

        # score_masks = (1 - attn_masks1) * -1e30
        # score_masks = score_masks.unsqueeze(dim=-1).unsqueeze(dim=-1)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        # attn_scores = attn_scores + score_masks
        attn_probs = nn.Softmax(dim=-1)(attn_scores)
        attn_probs = self.dropout(attn_probs)

        context_output = torch.matmul(attn_probs, v)
        context_output = context_output.permute(0, 2, 1, 3).contiguous()
        context_output = context_output.view(bs, -1, self.hidden_size)

        return context_output, attn_probs
    
class TextImageFusionModule(nn.Module):
    def __init__(self, hidden_dim, latent, num_heads=16):
        super(TextImageFusionModule, self).__init__()

        self.linear_text_to_image = nn.Linear(latent, latent * 2)  # 文本嵌入到图像空间的线性映射

        self.attention = CrossMultiHeadAttention(hidden_size=latent, num_heads=num_heads)

        self.feedforward = FeedForwardLayer(latent)

        self.ln1 = nn.LayerNorm(latent)
        self.ln2 = nn.LayerNorm(latent)

        self.vae = VAE(hidden_dim, latent)

    def forward(self, fusion_embeddings, embeddings, isImg=True):
        fusion_embeddings = self.vae.encoder(fusion_embeddings)
        embeddings = self.vae.encoder(embeddings)
        if isImg:
            batch_size, image_num, patch_num, hidden_dim = embeddings.size()

            # 将文本嵌入映射到图像空间
            fusion_embeddings_k, fusion_embeddings_v = torch.split(self.linear_text_to_image(fusion_embeddings).unsqueeze(1).expand(-1, image_num, -1, -1), hidden_dim, dim=-1)  # [batch, imgNum, length, hidden_dim]
            # 将文本嵌入扩展为 K 和 V
            K = torch.cat([fusion_embeddings_k, embeddings], dim=2).reshape(batch_size * image_num, -1, hidden_dim)  # [batch * image_num, length+patchNum, hidden_dim]
            V = torch.cat([fusion_embeddings_v, embeddings], dim=2).reshape(batch_size * image_num, -1, hidden_dim)

            # 将图像嵌入作为 Q
            Q = embeddings.view(batch_size * image_num, patch_num, hidden_dim)  # [batch * image_num, patch_num, hidden_dim]
        else:
            batch_size, image_num, hidden_dim = embeddings.size()

            fusion_embeddings_k, fusion_embeddings_v = torch.split(self.linear_text_to_image(fusion_embeddings), hidden_dim, dim=-1)  # [batch, length, hidden_dim]
            # 将文本嵌入扩展为 K 和 V
            K = torch.cat([fusion_embeddings_k.view(batch_size, -1, hidden_dim), embeddings], dim=1)  # [batch, length+imgNum, hidden_dim]
            V = torch.cat([fusion_embeddings_v.view(batch_size, -1, hidden_dim), embeddings], dim=1)

            Q = embeddings
        # 多头注意力
        attn_output, _ = self.attention(Q, K, V)  # 输出: [batch * image_num, patch_num, hidden_dim]

        # 残差连接和层归一化
        attn_output = self.ln1(attn_output + Q)

        # 前馈神经网络
        ff_output = self.feedforward(attn_output)

        # 残差连接和层归一化
        output = self.ln2(ff_output + attn_output)

        output = self.vae.decoder(output)
        return output
    
class VAE(nn.Module):
    def __init__(self, hidden_dim, latent):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, latent),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 256),
            nn.ReLU(True),
            nn.Linear(256, hidden_dim),
            nn.Sigmoid()  # 使用 Sigmoid 以确保输出在 [0, 1] 范围内
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class MissModalGenerate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.prefix_tokens = torch.arange(config.promptLen).long()
        self.prompt = torch.nn.Embedding(config.promptLen, 3 * config.bert_hid_size)
        self.conv_layers = nn.Conv1d(in_channels=config.bert_hid_size, out_channels=config.bert_hid_size, kernel_size=1)
        self.conv_layers_prompt = nn.ModuleDict({
            "0": nn.Conv1d(in_channels=config.bert_hid_size, out_channels=config.bert_hid_size, kernel_size=1),
            "1": nn.Conv1d(in_channels=config.bert_hid_size, out_channels=config.bert_hid_size, kernel_size=1),
            "2": nn.Conv1d(in_channels=config.bert_hid_size, out_channels=config.bert_hid_size, kernel_size=1),
            })
        self.relu = nn.ReLU(inplace=False)

    def forward(self, modal, is_text=True):

        bs, _, hidden_dim = modal.shape
        prefix_tokens = self.prefix_tokens.to(self.config.device).unsqueeze(0).expand(bs, -1)

        prompt = self.prompt(prefix_tokens).view(3, bs, -1, hidden_dim)

        # prompt: 3 * bs * length * hidden
        conv_modal = self.conv_layers(modal.permute(0, 2, 1)) # bs  * hidden * length
        conv_modal = self.relu(conv_modal)
        conv_outputs = []
        for i in range(3):
            conv_output = self.conv_layers_prompt[str(i)](torch.cat([prompt[i, :, :, :].permute(0, 2, 1), conv_modal], dim=-1)).permute(0, 2, 1)
            conv_output = self.relu(conv_output)
            if not is_text:
                init_len = conv_output.shape[1]
                miss_len = (init_len // 197 + 1) * 197
                padding = torch.zeros((bs, (miss_len-init_len), hidden_dim)).to(self.config.device)
                conv_output = torch.cat([conv_output, padding], dim=1)
            conv_outputs.append(conv_output)
        return tuple(conv_outputs)
    

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.config = config
        self.lstm_hid_size = config.lstm_hid_size
        self.height=config.height
        self.width=config.width
        self.bert_hid_size=config.bert_hid_size

        self.bert = AutoModel.from_pretrained(
                    config.bert_name, 
                    max_length=config.max_sequence_length,
                    output_hidden_states=True
                )

        self.vit = ViTModel.from_pretrained(config.vit_name, output_hidden_states=True, use_mask_token=True)
        self.ttt = 0
        # self.encoder = nn.LSTM(config.bert_hid_size, config.lstm_hid_size // 2, num_layers=1, batch_first=True, bidirectional=True)
        if config.is_h:
            self.attention = nn.ModuleDict({
                "True-True-0": features_attention(config, feature_attention=False),
                "True-True-1": features_attention(config, feature_attention=False),
                "True-True-2": features_attention(config, feature_attention=False),
                "True-False-0": TextImageFusionModule(config.bert_hid_size, config.latent),
                "True-False-1": TextImageFusionModule(config.bert_hid_size, config.latent),
                "True-False-2": TextImageFusionModule(config.bert_hid_size, config.latent),
                "False-False-0": TextImageFusionModule(config.bert_hid_size, config.latent),
                "False-False-1": TextImageFusionModule(config.bert_hid_size, config.latent),
                "False-False-2": TextImageFusionModule(config.bert_hid_size, config.latent)
                })
        else:
            config.is_decoder = False
            self.attention = BertAttention(config)
        
        self.FFN = nn.ModuleDict({
            "False": FeedForwardLayer(config.bert_hid_size),
            "True": FeedForwardLayer(config.bert_hid_size)
            })
        
        # self.encoder = nn.ModuleDict({
        #     "False": nn.LSTM(config.bert_hid_size, config.lstm_hid_size // 2, num_layers=1, batch_first=True, bidirectional=True),
        #     "True": nn.LSTM(config.bert_hid_size, config.lstm_hid_size // 2, num_layers=1, batch_first=True, bidirectional=True)
        #     })
        # self.attention_feature = features_attention(config)
        # self.attention = features_attention(config, feature_attention=False)

        self.ent_output = nn.Linear(config.lstm_hid_size, config.ent_label_num)
        torch.nn.init.orthogonal_(self.ent_output.weight, gain=1)

        self.ent_dropout = nn.Dropout(config.dropout)

        self.link_linear_1 = nn.Linear(config.lstm_hid_size, config.lstm_hid_size)
        torch.nn.init.orthogonal_(self.link_linear_1.weight, gain=1)
        self.link_linear_2 = nn.Linear(config.lstm_hid_size, config.lstm_hid_size)
        torch.nn.init.orthogonal_(self.link_linear_2.weight, gain=1)

        self.link_dropout = nn.Dropout(config.dropout)

        self.rel_output = nn.Linear(config.lstm_hid_size, config.rel_label_num)
        torch.nn.init.orthogonal_(self.rel_output.weight, gain=1)

        self.rel_linear = nn.Linear(config.lstm_hid_size, config.lstm_hid_size)
        torch.nn.init.orthogonal_(self.rel_linear.weight, gain=1)

        self.rel_dropout = nn.Dropout(config.dropout)
        
        self.bili = nn.ModuleDict({
            "link": torch.nn.Bilinear(config.lstm_hid_size, config.lstm_hid_size, 2),
            "rel": torch.nn.Bilinear(config.lstm_hid_size, config.lstm_hid_size, config.rel_label_num),
            })
        
        self.ln = nn.LayerNorm(config.lstm_hid_size)
        self.frame_pos_embedding = nn.Parameter(torch.zeros(1, config.maxImgNum, config.lstm_hid_size))
        self.text_fusion_weight_piece = nn.Parameter(torch.randn(4, 1))
        self.text_fusion_weight_token = nn.Parameter(torch.randn(4, 1))
        self.img_fusion_weight = nn.Parameter(torch.randn(4, 1))

        # self.bili = torch.nn.Bilinear(config.lstm_hid_size, config.lstm_hid_size, 2)
        self.crf = CRF(config.ent_label_num, batch_first=True)
        self.loss_fct = CrossEntropyLoss()

        self.classification_img = nn.Linear(config.lstm_hid_size, config.ent_label_num // 2 - 1 + 1) # 减少一个TIME类，增加一个None类
        self.bbox_head = nn.Linear(config.lstm_hid_size, 4)  # Bounding box coordinates

        self.ground_l = nn.Linear(config.lstm_hid_size, config.lstm_hid_size*config.maxGroundNum)

        self.cluster_threshold = config.cluster_threshold
        self.current_bs, self.lstm_embedding, self.alpha = 0, None, config.alpha
        self.docs, self.ent_text, self.link_text, self.rel_text = None, [], [], []

        self.imgGen = MissModalGenerate(config)
        self.textGen = MissModalGenerate(config)

    def get_pair_tensor(self, ent_1_info, ent_2_info, bs, task="link"):
        ent_1_s, ent_1_e = int(ent_1_info.split(":")[0]), int(ent_1_info.split(":")[1])
        ent_2_s, ent_2_e = int(ent_2_info.split(":")[0]), int(ent_2_info.split(":")[1])
        ent_1 = torch.mean(self.lstm_embedding[bs, ent_1_s:ent_1_e, :], dim=0, keepdim=True)
        ent_2 = torch.mean(self.lstm_embedding[bs, ent_2_s:ent_2_e, :], dim=0, keepdim=True)
        pair_tensor = self.bili[task](ent_1, ent_2) #1 * rel_num+1
        pair_tensor = self.rel_dropout(pair_tensor)
        return pair_tensor

    def get_ent_pair_tensor(self, ent_rel, max_len, task="link"):
        batch_res = []
        for bs, batch_ent_rel in enumerate(ent_rel):
            temp_ent_rel = []
            for pair in batch_ent_rel:
                ent_1_info = pair.split("-")[0]
                ent_2_info = pair.split("-")[1]
                temp_ent_rel.append(self.get_pair_tensor(ent_1_info, ent_2_info, bs, task=task))
            batch_len = len(temp_ent_rel)
            assert max_len >= batch_len
            if task == "link":
                temp_ent_rel.extend([torch.zeros(1, 2).to(self.config.device)] * (max_len - batch_len))
            else:
                temp_ent_rel.extend([torch.zeros(1, self.config.rel_label_num).to(self.config.device)] * (max_len - batch_len))
            if temp_ent_rel:
                temp_ent_rel = torch.cat(temp_ent_rel, dim=0)
            else:
                if task == "link":
                    temp_ent_rel = torch.zeros((1, 2)).to(self.config.device)
                else:
                    temp_ent_rel = torch.zeros((1, self.config.rel_label_num)).to(self.config.device)
            batch_res.append(temp_ent_rel)
        logits = torch.stack(batch_res, dim=0)

        return logits

    """
        1. Calculate accuracy, recall, and F1.
        2. preds and golds are list, type of every item is String, example prerds=["start index:end index-entity type", ...]
    """
    def ent_prc(self, preds, golds): 
        p, r, c = 0, 0, 0
        r += len(golds)
        p += len(preds)
        error_ent_be, error_ent_te = 0, 0
        golds_dic = {g.split("##")[0]:g.split("##")[1] for g in golds}
        for pred in preds:
            if pred in golds:
                c += 1
            else:
                if pred.split("##")[0] in golds_dic:
                    error_ent_te += 1
                else:
                    error_ent_be += 1
        return p, r, c, error_ent_be, error_ent_te
    
    def ent_link_decode(self, ent_pair_rel, ent_link):
        batch_ceaf_c, batch_ceaf_p, batch_ceaf_r, batch_muc_c, batch_muc_p, batch_muc_r, batch_b3_c_p, batch_b3_c_r, batch_b3_p, batch_b3_r = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        error_link_p, error_link_ee, error_link_ne = 0, 0, 0
        for bs, batch_gold in enumerate(ent_link):
            pred = []
            for pair in ent_pair_rel[bs]:
                ent_1_info = pair.split("-")[0]
                ent_2_info = pair.split("-")[1]
                pair_tensor = self.get_pair_tensor(ent_1_info, ent_2_info, bs, task="link")
                pair_type = torch.argmax(pair_tensor, dim=1)
                if int(pair_type):
                    pred.append(pair.split("-"))
            batch_pred = merge_overlapping_sublists(pred)
            error_link_p += len(batch_pred)

            quchong = []
            for pred in batch_pred:
                max_intersection = []
                max_length = 0
                for gold in batch_gold:
                    if gold not in quchong:
                        # 计算交集
                        intersection = set(gold) & set(pred)
                        # 检查交集的大小
                        if len(intersection) > max_length:
                            max_length = len(intersection)
                            max_intersection = [intersection, gold]
                if max_intersection:
                    quchong.append(max_intersection[-1])
                    intersection = max_intersection[0]
                    if len(intersection) == len(pred) and len(intersection) == len(max_intersection[-1]):
                        pass
                    elif len(intersection) == len(pred) and len(intersection) != len(max_intersection[-1]):
                        error_link_ne += 1
                    else:
                        error_link_ee += 1
                else:
                    error_link_ee += 1
            muc_c, muc_p, muc_r = muc(batch_pred, batch_gold)
            ceaf_c, ceaf_p, ceaf_r = ceaf(batch_pred, batch_gold)
            b3_c_p, b3_c_r, b3_p, b3_r = b3(batch_pred, batch_gold)

            batch_muc_c += muc_c
            batch_muc_p += muc_p
            batch_muc_r += muc_r
            batch_ceaf_c += ceaf_c
            batch_ceaf_p += ceaf_p
            batch_ceaf_r += ceaf_r
            batch_b3_c_p += b3_c_p
            batch_b3_c_r += b3_c_r
            batch_b3_p += b3_p
            batch_b3_r += b3_r

        return batch_ceaf_c, batch_ceaf_p, batch_ceaf_r, batch_muc_c, batch_muc_p, batch_muc_r, batch_b3_c_p, batch_b3_c_r, batch_b3_p, batch_b3_r, error_link_p, error_link_ne, error_link_ee
    
    def ent_rel_decode(self, all_ent_pair_base_link_list, ent_rel):
        batch_p, batch_c, batch_r = 0, 0, 0
        all_link_pair_type = []
        rel_id2type = self.config.vocab.id2label_rel
        for bs, batch_ent_pair_base_link_dic in enumerate(all_ent_pair_base_link_list):
            ent_rel_golds = []
            for ent_rel_gold in ent_rel[bs]:
                if ent_rel_gold.split(":")[-1] != "None":
                    ent_rel_golds.append(ent_rel_gold)
            batch_r += len(ent_rel_golds)
            batch_link_pair_type = {} 
            for link_pair, link_ent_pair in batch_ent_pair_base_link_dic.items():
                pair_type_temp = {}
                for pair in link_ent_pair:
                    ent_1_info = pair.split("-")[0]
                    ent_2_info = pair.split("-")[1]
                    ent_1_type, ent_2_type = ent_1_info.split(":")[2], ent_2_info.split(":")[2]
                    ent_1_text, ent_2_text = ent_1_info.split(":")[-1], ent_2_info.split(":")[-1]
                    if f"{ent_1_type}-{ent_2_type}" == "TIME-TIME" or ent_1_text == ent_2_text:
                            continue
                    pair_tensor = self.get_pair_tensor(ent_1_info, ent_2_info, bs, task="rel")
                    for type_id, rel_type in rel_id2type.items():
                        if f"{ent_1_type}-{ent_2_type}" not in rel_type and f"{ent_2_type}-{ent_1_type}" not in rel_type and rel_type != "None":
                            pair_tensor[:, type_id] = -100
                    pair_type = torch.argmax(pair_tensor, dim=1)
                    pair_type = rel_id2type[int(pair_type)]
                    if pair_type != "None":
                        batch_p += 1
                        if f"{ent_1_info}-{ent_2_info}:{pair_type}" in ent_rel_golds or f"{ent_2_info}-{ent_1_info}:{pair_type}" in ent_rel_golds:
                            batch_c += 1
                            try:
                                ent_rel_golds.remove(f"{ent_1_info}-{ent_2_info}:{pair_type}")
                            except:
                                ent_rel_golds.remove(f"{ent_2_info}-{ent_1_info}:{pair_type}")
                    if pair_type not in pair_type_temp:
                        pair_type_temp[pair_type] = 1
                    else:
                        pair_type_temp[pair_type] += 1
                if pair_type_temp:
                    batch_link_pair_type[link_pair] = sorted(pair_type_temp.items(), key=lambda x:x[1],reverse=True)[0][0]
                else:
                    batch_link_pair_type[link_pair] = "None"
            all_link_pair_type.append(batch_link_pair_type)
        return batch_p, batch_c, batch_r, all_link_pair_type
                
    """
        1. Obtaining Entity Chains through Hierarchical Clustering and Entity Chain Pairs Embedding.
        2. entities is list, type of every item is String, example entities=["ent1 start index:ent1 end index_ent2 start index:ent2 end index_...", ...].
        3. batch_link_pair_tensor is dict, example batch_link_pair_tensor={Entity Chain Pairs:Entity Chain Pairs Embedding}, the type of key is String and the type of value is Tensor.
    """
    def clustering(self, entities):
        while True:
            batch_ent_pair_base_link_dic = {}
            if len(entities) <= 1:
                break
            entity_pair = get_pair(entities)
            s_dic = {}
            for pair in entity_pair:
                m1, m2 = pair.split("-")
                ent_list_1 = m1.split("_")
                ent_list_2 = m2.split("_")
                l1 = len(ent_list_1)
                l2 = len(ent_list_2)
                l = l1 + l2
                m1_tensor = torch.mean(torch.cat([self.lstm_embedding[self.current_bs, int(m.split(":")[0]):int(m.split(":")[1]), :] for m in m1.split("_")], dim=0), dim=0, keepdim=True)
                m2_tensor = torch.mean(torch.cat([self.lstm_embedding[self.current_bs, int(m.split(":")[0]):int(m.split(":")[1]), :] for m in m2.split("_")], dim=0), dim=0, keepdim=True)
                batch_ent_pair_base_link_dic[pair] = get_pair(ent_list_1, ent_list_2)
                s = float(torch.cosine_similarity(m1_tensor, m2_tensor)) * 2 / l
                s_dic[pair] = s
            sorted_s = sorted(s_dic.items(), key=lambda x:x[-1], reverse=True)
            max_pair, s = sorted_s[0]
            if s > self.cluster_threshold:
                entities.remove(max_pair.split("-")[0])
                entities.remove(max_pair.split("-")[-1])
                entities.append(f"{max_pair.replace('-', '_')}")
            else:
                break
        return entities, batch_ent_pair_base_link_dic
    
    """
        1. Obtain entity chains and the embedding of entity chain pairs for all batches.
        2. batch_type_ind is dict, example batch_link_pair_tensor={Entity type:Entity Chain list}, the type of key is String and the type of value is List.
        3. batch_link_pair_tensor is dict, example batch_link_pair_tensor={Entity Chain Pairs:Entity Chain Pairs Embedding}, the type of key is String and the type of value is Tensor.
    """
    def ent_gold2typeDic(self, ent_gold_list):
        batch_type_ind_temp = {}
        for item in ent_gold_list:
            ent_type = item.split(":")[2]
            if ent_type not in batch_type_ind_temp:
                batch_type_ind_temp[ent_type] = [item]
            else:
                batch_type_ind_temp[ent_type].append(item)
        return batch_type_ind_temp
    
    def gold_link2ent_pairDic(self, batch_gold_links):
        batch_ent_pair_base_gold_link_dic = {}
        for ent_list_1 in batch_gold_links:
            for ent_list_2 in batch_gold_links:
                pair = f'{"_".join(ent_list_1)}-{"_".join(ent_list_2)}'
                batch_ent_pair_base_gold_link_dic[pair] = get_pair(ent_list_1, ent_list_2)
        return batch_ent_pair_base_gold_link_dic

    def get_batch_link(self, batch_type_ind, bacth_link_outputs=None, is_gold=False):
        batch_ent_pair_base_gold_link_dic = {}
        if is_gold:
            batch_type_ind = self.ent_gold2typeDic(batch_type_ind)
            batch_ent_pair_base_gold_link_dic = self.gold_link2ent_pairDic(bacth_link_outputs)
        type_link, batch_ent_pair_base_link_dic = {}, {}
        for ent_type, all_ind in batch_type_ind.items():
            link_res, batch_ent_pair_base_link_dic = self.clustering(all_ind)
            if ent_type not in type_link:
                type_link[ent_type] = link_res
        return type_link, batch_ent_pair_base_link_dic, batch_ent_pair_base_gold_link_dic
    
    def ent2id(self, bio_list):
        temp = []
        ent_res = []
        for ind, i in enumerate(bio_list):
            if i % 2 != 0:
                if temp:
                    ent_res.append(f"{temp[1]}-{temp[-1]}##{temp[0]}")
                    temp = []
                temp.append(i)
                temp.append(ind)
            elif i != 0 and i % 2 == 0:
                if temp and i == temp[0] + 1:
                    temp.append(ind)
            elif i == 0:
                if temp:
                    ent_res.append(f"{temp[1]}-{temp[-1]}##{temp[0]}")
                    temp = []
        if temp:
            ent_res.append(f"{temp[1]}-{temp[-1]}##{temp[0]}")
            temp = []
        return ent_res

    """
        1. Decoding BIO sequences and evaluating entity recognition results.
        2. ent_output and ent_outputs are list, type of every item is String, example ent_output=["O", "O", "B-PER"...], ent_outputs=[["start index:end index-entity type", ...], ["start index:end index-entity type", ...]]
    """
    def ent_decode(self, ent_output, ent_outputs, link_outputs, grid_labels_ent):
        id2label_ent = self.config.vocab.id2label_ent
        batch_p, batch_c, batch_r = 0, 0, 0
        batch_error_ent_te, batch_error_ent_be = 0, 0
        all_link, all_ent_pair_base_link_list, all_link_base_gold_ent, all_ent_pair_base_gold_link_list = [], [], [], []
        for bs, batch_ent_output in enumerate(ent_output):
            self.current_bs = bs
            ind_ent, batch_ent_preds = {}, []
            temp = []
            for ind, label_id in enumerate(batch_ent_output):
                bio = id2label_ent[int(label_id)]
                if "-" in bio:
                    ent_type = bio.split("-")[-1]
                    if "B-" in bio:
                        if temp:
                            if ent_type not in ind_ent:
                                ind_ent[ent_type] = [f"{temp[1]}:{temp[-1]+1}:{ent_type}:{''.join(self.docs[bs][temp[1]:temp[-1]+1]).replace('-', '——')}"]
                            else:
                                ind_ent[ent_type].append(f"{temp[1]}:{temp[-1]+1}:{ent_type}:{''.join(self.docs[bs][temp[1]:temp[-1]+1]).replace('-', '——')}")
                            batch_ent_preds.append(f"{temp[1]}:{temp[-1]+1}:{ent_type}:{''.join(self.docs[bs][temp[1]:temp[-1]+1]).replace('-', '——')}")
                            temp = []
                        temp.extend([ent_type, ind])
                    else:
                        if temp:
                            temp.append(ind)
                else:
                    if temp:
                        if ent_type not in ind_ent:
                            ind_ent[ent_type] = [f"{temp[1]}:{temp[-1]+1}:{ent_type}:{''.join(self.docs[bs][temp[1]:temp[-1]+1]).replace('-', '——')}"]
                        else:
                            ind_ent[ent_type].append(f"{temp[1]}:{temp[-1]+1}:{ent_type}:{''.join(self.docs[bs][temp[1]:temp[-1]+1]).replace('-', '——')}")
                        batch_ent_preds.append(f"{temp[1]}:{temp[-1]+1}:{ent_type}:{''.join(self.docs[bs][temp[1]:temp[-1]+1]).replace('-', '——')}")
                        temp = []
            if temp:
                if ent_type not in ind_ent:
                    ind_ent[ent_type] = [f"{temp[1]}:{temp[-1]+1}:{ent_type}:{''.join(self.docs[bs][temp[1]:temp[-1]+1]).replace('-', '——')}"]
                else:
                    ind_ent[ent_type].append(f"{temp[1]}:{temp[-1]+1}:{ent_type}:{''.join(self.docs[bs][temp[1]:temp[-1]+1]).replace('-', '——')}")
                batch_ent_preds.append(f"{temp[1]}:{temp[-1]+1}:{ent_type}:{''.join(self.docs[bs][temp[1]:temp[-1]+1]).replace('-', '——')}")
                temp = []
            p, r, c, error_ent_be, error_ent_te = self.ent_prc(self.ent2id(batch_ent_output), self.ent2id(grid_labels_ent[self.current_bs]))
            batch_p += p
            batch_r += r
            batch_c += c
            batch_error_ent_be += error_ent_be
            batch_error_ent_te += error_ent_te
            batch_link_res, batch_ent_pair_base_link_dic, _ = self.get_batch_link(ind_ent)
            batch_link_res_base_gold_ent, _, batch_ent_pair_base_gold_link_dic = self.get_batch_link(ent_outputs[self.current_bs], link_outputs[self.current_bs], is_gold=True)
            all_link.append(batch_link_res)
            all_link_base_gold_ent.append(batch_link_res_base_gold_ent)
            all_ent_pair_base_link_list.append(batch_ent_pair_base_link_dic)
            all_ent_pair_base_gold_link_list.append(batch_ent_pair_base_gold_link_dic)
        return batch_p, batch_r, batch_c, batch_error_ent_be, batch_error_ent_te, all_link, all_ent_pair_base_link_list, all_link_base_gold_ent, all_ent_pair_base_gold_link_list
    

    def link_rel_preds(self, all_link_pair_type, rel_outputs):
        copy_rel_outputs = copy.deepcopy(rel_outputs)
        link_rel_p, link_rel_c, link_rel_r = 0, 0, 0
        error_rel_te, error_rel_ne = 0, 0
        for bs, batch_link_pair_type in enumerate(all_link_pair_type):
            batch_rel_golds = copy_rel_outputs[bs]
            link_rel_r += batch_rel_golds["num"]
            del batch_rel_golds["num"]
            for pair, pair_type in batch_link_pair_type.items():
                if pair_type != "None":
                    link_1 = pair.split("-")[0]
                    link_2 = pair.split("-")[1]
                    pred_all_ent_pair = set(get_pair(link_1.split("_"), link_2.split("_")))
                    link_rel_p += 1
                    if pair_type in batch_rel_golds:
                        is_error = True
                        for ind, all_golds_link_list in enumerate(batch_rel_golds[pair_type]):
                            if pred_all_ent_pair & all_golds_link_list:
                                link_rel_c += 1
                                batch_rel_golds[pair_type].pop(ind)
                                is_error = False
                                break
                        if is_error:
                            error_rel_ne += 1
                    else:
                        is_error = True
                        for pair_type in batch_rel_golds:
                            for ind, all_golds_link_list in enumerate(batch_rel_golds[pair_type]):
                                if pred_all_ent_pair & all_golds_link_list:
                                    is_error = False
                                    break
                        if is_error:
                            error_rel_ne += 1
                        else:
                            error_rel_te += 1
                            
        return link_rel_p, link_rel_c, link_rel_r, error_rel_te, error_rel_ne

    def linkEvluate(self, preds, golds):
        batch_ceaf_c, batch_ceaf_p, batch_ceaf_r, batch_muc_c, batch_muc_p, batch_muc_r, batch_b3_c_p, batch_b3_c_r, batch_b3_p, batch_b3_r = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        for batch in range(preds.__len__()):
            batch_pred = []
            for link_list in preds[batch].values():
                batch_pred.extend([link.split("_") for link in link_list])
            batch_gold = golds[batch]

            muc_c, muc_p, muc_r = muc(batch_pred, batch_gold)
            ceaf_c, ceaf_p, ceaf_r = ceaf(batch_pred, batch_gold)
            b3_c_p, b3_c_r, b3_p, b3_r = b3(batch_pred, batch_gold)

            batch_muc_c += muc_c
            batch_muc_p += muc_p
            batch_muc_r += muc_r
            batch_ceaf_c += ceaf_c
            batch_ceaf_p += ceaf_p
            batch_ceaf_r += ceaf_r
            batch_b3_c_p += b3_c_p
            batch_b3_c_r += b3_c_r
            batch_b3_p += b3_p
            batch_b3_r += b3_r

        return batch_ceaf_c, batch_ceaf_p, batch_ceaf_r, batch_muc_c, batch_muc_p, batch_muc_r, batch_b3_c_p, batch_b3_c_r, batch_b3_p, batch_b3_r

    def get_fusion_feature(self, hidden_outputs, embs_last, hidden_outputs_isImg=True, is_piece=False, is_miss=False):
        if not is_miss:
            # hidden_outputs: bs * length * hidden or bs * imgNum * patchNum * hidden
            span = len(hidden_outputs) // 3
            features = []
            for i in range(len(hidden_outputs)//span):
                outputs = hidden_outputs[i*span:(i+1)*span]
                outputs = torch.stack(outputs, dim=0)
                if hidden_outputs_isImg: # text
                    outputs = torch.mean(outputs, dim=0).view(self.bs, -1, self.bert_hid_size)
                    if is_piece:
                        inputs_attention_embs = self.attention[f"{hidden_outputs_isImg}-{is_piece}-{i}"](embs_last, outputs) #bs*len*768
                    else:
                        inputs_attention_embs = self.attention[f"{hidden_outputs_isImg}-{is_piece}-{i}"](outputs, embs_last, isImg=False) #bs*len*768
                    packed_outs = self.FFN[f"{hidden_outputs_isImg}"](inputs_attention_embs) # bs*len*lstm_hidden   
                else:
                    outputs = torch.mean(outputs, dim=0)
                    embs_last = embs_last.view(self.bs, self.imgNum, -1, self.bert_hid_size)
                    packed_outs = self.attention[f"{hidden_outputs_isImg}-{is_piece}-{i}"](outputs, embs_last) #bs*len*768

                features.append(packed_outs)
            return tuple(features)
        else:
            for i, outputs in enumerate(hidden_outputs):
                if hidden_outputs_isImg: # text
                    outputs = torch.mean(outputs, dim=0).view(self.bs, -1, self.bert_hid_size)
                    if is_piece:
                        inputs_attention_embs = self.attention[f"{hidden_outputs_isImg}-{is_piece}-{i}"](embs_last, outputs) #bs*len*768
                    else:
                        inputs_attention_embs = self.attention[f"{hidden_outputs_isImg}-{is_piece}-{i}"](outputs, embs_last, isImg=False) #bs*len*768
                    packed_outs = self.FFN[f"{hidden_outputs_isImg}"](inputs_attention_embs) # bs*len*lstm_hidden   
                else:
                    outputs = torch.mean(outputs, dim=0)
                    embs_last = embs_last.view(self.bs, self.imgNum, -1, self.bert_hid_size)
                    packed_outs = self.attention[f"{hidden_outputs_isImg}-{is_piece}-{i}"](outputs, embs_last) #bs*len*768
                features.append(packed_outs)
            return tuple(features)
    
    def yolo_to_corners(self, box):
        """
        将 YOLO 格式 (x_center, y_center, width, height) 转换为
        (x_min, y_min, x_max, y_max)
        """
        x_center, y_center, width, height = box
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2
        return torch.tensor([x_min, y_min, x_max, y_max])

    def calculate_iou(self, box1, box2):
        """
        box1, box2: YOLO 格式的边框 (x_center, y_center, width, height)
        """
        # 将 YOLO 格式转换为 (x_min, y_min, x_max, y_max)
        box1_corners = self.yolo_to_corners(box1)
        box2_corners = self.yolo_to_corners(box2)

        # 获取交集区域的坐标
        x_inter_min = torch.max(box1_corners[0], box2_corners[0])
        y_inter_min = torch.max(box1_corners[1], box2_corners[1])
        x_inter_max = torch.min(box1_corners[2], box2_corners[2])
        y_inter_max = torch.min(box1_corners[3], box2_corners[3])

        # 计算交集区域的宽度和高度
        inter_width = torch.clamp(x_inter_max - x_inter_min, min=0)
        inter_height = torch.clamp(y_inter_max - y_inter_min, min=0)

        # 计算交集面积
        inter_area = inter_width * inter_height

        # 计算每个边框的面积
        area1 = (box1_corners[2] - box1_corners[0]) * (box1_corners[3] - box1_corners[1])
        area2 = (box2_corners[2] - box2_corners[0]) * (box2_corners[3] - box2_corners[1])

        # 计算并集面积
        union_area = area1 + area2 - inter_area

        # 计算 IoU
        iou = inter_area / union_area

        return iou

    def merge_overlapping_regions(self, tensor, iou_threshold=0.9):
        """合并重叠区域，将类别和边框置为0"""
        m, n, _ = tensor.shape
        for i in range(m):  # 遍历每张图片
            for j in range(n):
                if tensor[i, j, 0] == 0:  # 已经被置0的区域跳过
                    continue
                for k in range(j + 1, n):
                    if tensor[i, k, 0] == 0:
                        continue
                    # 计算 j 和 k 区域的 IoU
                    iou = self.calculate_iou(tensor[i, j, 1:], tensor[i, k, 1:])
                    if iou > iou_threshold:
                        # 如果重叠度较高，选择将其中一个区域置为0
                        tensor[i, k, :] = 0  # 将类别和边框置为0
        return tensor[:, :, 0], tensor[:, :, 1:]
    
    def ground_encode(self, preds_class, preds_bourd, golds):
        ground_p, ground_r, ground_c = 0, 0, 0
        error_gro_be, error_gro_te = 0, 0
        for pred_class, pred_bourd, gold in zip(preds_class, preds_bourd, golds):
            ground_r += int(torch.count_nonzero(gold[:, :, 0]))
            pred_class, pred_bourd = self.merge_overlapping_regions(torch.cat([pred_class.unsqueeze(-1), pred_bourd], dim=-1))
            for img_id in range(gold.shape[0]):
                per_class = pred_class[img_id, :]
                per_bourd = pred_bourd[img_id, :, :]
                # 去除重叠度高的预测值
                per_gold = gold[img_id, :, :]
                for i, p in enumerate(per_class):
                    per_bourd_class = int(p)
                    per_bourd_bourd = per_bourd[i]
                    if per_bourd_class:
                        if_type_error = True
                        if_bourd_error = True
                        for g in range(per_gold.shape[0]):
                            gold_class = int(per_gold[g, 0])
                            gold_bourd = per_gold[g, 1:]
                            if gold_class:
                                iou = self.calculate_iou(per_bourd_bourd, gold_bourd)
                                if gold_class == per_bourd_class:
                                    if_type_error = False
                                    if float(iou) > 0.5:
                                        if_bourd_error = False
                                        ground_c += 1
                                        per_gold[g, :] = 0
                                else:
                                    if float(iou) > 0.5:
                                        if_bourd_error = False
                        if if_type_error and if_bourd_error:
                            error_gro_be += 0.5
                            error_gro_te += 0.5
                        elif not if_type_error and if_bourd_error:
                            error_gro_be += 1
                        elif if_type_error and not if_bourd_error:
                            error_gro_te += 1
        return ground_p, ground_r, ground_c, error_gro_be, error_gro_te
    
    def calu_img_loss(self, class_outputs, bbox_outputs, target_classes, target_bboxes):
        # Compute classification loss
        cls_loss = F.cross_entropy(class_outputs, target_classes)
        
        # Compute bounding box regression loss (L1 loss)
        bbox_loss = F.l1_loss(bbox_outputs, target_bboxes)

        total_loss = cls_loss + bbox_loss
        return total_loss
    
    def piece2word(self, piece_embs, pieces2word):
        length = pieces2word.size(1)

        min_value = torch.min(piece_embs).item()

        # Max pooling word representations from pieces
        _piece_embs = piece_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _piece_embs = torch.masked_fill(_piece_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_embs, _ = torch.max(_piece_embs, dim=2)
        return word_embs
    
    def calu_ent_loss(self, golds, preds):
        active_labels_ent = golds.cpu().numpy()
        active_labels_ent[(active_labels_ent == 0)] = -1
        active_labels_ent[(active_labels_ent == -100)] = 0
        ent_labels_mask = torch.from_numpy(active_labels_ent).bool().to(self.config.device)
        active_labels_ent[(active_labels_ent == -1)] = 0
        new_l = torch.from_numpy(active_labels_ent).long().to(self.config.device)
        loss = -self.crf(preds, new_l, mask=ent_labels_mask, reduction='mean')
        return loss
    
    def miss_modal(self, hidden_states, miss, bs, i, is_text=True):
        hidden_states = [h.clone().view(bs, -1, self.bert_hid_size) for h in hidden_states]
        init_len = hidden_states[0].shape[1]
        miss_len = miss[0].shape[1]
        if init_len >= miss_len:
            for j in range(len(hidden_states)):
                hidden_states[j][i, :miss_len, :] =  miss[(j-1)//4][i, :, :]
        else:
            padding = torch.zeros((bs, (miss_len-init_len), self.bert_hid_size)).to(self.config.device)
            for j in range(len(hidden_states)):
                hidden_states[j] = torch.cat([hidden_states[j], padding], dim=1)
                hidden_states[j][i, :, :] =  miss[(j-1)//4][i, :, :]
        if not is_text:
            hidden_states = [h.clone().view(-1, 197, self.bert_hid_size) for h in hidden_states]
        return hidden_states
    
    def forward(self, bert_inputs, bert_inputs_token, bert_inputs_mask, ent_output_ids, grid_labels_ent, ent_output_mask, ent_link_ids, ent_link_mask, ent_rel_ids, ent_rel_mask, images, grid_labels_img, pieces2word, ent_outputs, link_outputs, rel_outputs, docs, ent_link, ent_rel, ent_pair_rel, dataType, is_train=True):
        """
            word embdding
        """
        bs, imgNum, _, _, _ = images.shape
        self.bs = bs
        self.imgNum = imgNum

        images = images.view(-1, 3, self.height, self.width)
        img_embs = self.vit(images) # bs * imgNum , patchNum , hidden

        self.docs = docs
        
        inputs_embs = self.bert(bert_inputs, bert_inputs_mask) # bs*len*768
        
        inputs_embs_token = self.bert(bert_inputs_token, attention_mask=bert_inputs_token.ne(0).float()) # bs*len*768

        img_embs_last = img_embs.last_hidden_state.view(self.bs, -1, self.bert_hid_size)

        word_embs_last = inputs_embs.last_hidden_state

        word_embs_last_token = inputs_embs_token.last_hidden_state

        img_embs_hidden_states = img_embs.hidden_states
        text_embs_hidden_states = inputs_embs_token.hidden_states

        if not self.config.is_miss:
            miss_img_embs_hidden_states = self.imgGen(word_embs_last, is_text=False)
            miss_text_embs_hidden_states = self.textGen(img_embs_last)
            for i, data_type in enumerate(dataType):
                if data_type == 1:
                    text_embs_hidden_states = self.miss_modal(text_embs_hidden_states, miss_text_embs_hidden_states, bs, i)
                elif data_type == 2:
                    img_embs_hidden_states = self.miss_modal(img_embs_hidden_states, miss_img_embs_hidden_states, bs, i, is_text=False)

        if self.config.is_h:
            text_l_piece, text_m_piece, text_h_piece = self.get_fusion_feature(img_embs_hidden_states, word_embs_last, hidden_outputs_isImg=True, is_piece=True)


            text_l_token, text_m_token, text_h_token = self.get_fusion_feature(img_embs_hidden_states, self.piece2word(word_embs_last_token, pieces2word), hidden_outputs_isImg=True, is_piece=False)


            text_fusion_piece = torch.einsum("asld,ab->bsld", torch.stack([word_embs_last, text_l_piece, text_m_piece, text_h_piece], dim=0), self.text_fusion_weight_piece).squeeze(0) # bs,length,hidden
            # text_fusion_piece = (text_l_piece+text_m_piece+text_h_piece)/3

            text_fusion_token = torch.einsum("asld,ab->bsld", torch.stack([self.piece2word(word_embs_last_token, pieces2word), text_l_token, text_m_token, text_h_token], dim=0), self.text_fusion_weight_token).squeeze(0) # bs,length,hidden

            if self.config == "zh":
                self.lstm_embedding = text_fusion_piece
            else:
                self.lstm_embedding = text_fusion_token
            
            img_l, img_m, img_h = [f.view(self.bs, imgNum, -1, self.lstm_hid_size) for f in self.get_fusion_feature(text_embs_hidden_states, img_embs_last, hidden_outputs_isImg=False, is_piece=False)]

            img_fusion = torch.einsum("asnld,ab->bsnld", torch.stack([img_embs_last.view(self.bs, imgNum, -1, self.bert_hid_size), img_l, img_m, img_h], dim=0), self.img_fusion_weight).squeeze(0) # bs,imgNum,patchNum,hidden
            # img_fusion = (img_l+img_m+img_h)/3
            
            img_fusion = self.ln(torch.sum(img_fusion, dim=2))
        else:
            img_fusion = self.ln(torch.sum(img_embs_hidden_states[-1].view(bs, imgNum, -1, self.bert_hid_size), dim=2))
            text = inputs_embs.last_hidden_state
            text_fusion_token = self.piece2word(inputs_embs_token.last_hidden_state, pieces2word)
            self.lstm_embedding = text

        img_fusion = self.ground_l(img_fusion + self.frame_pos_embedding[:, :imgNum, :]) # bs, imgNum, hidden*ground_num_max

        img_fusion = torch.stack(torch.split(img_fusion, self.lstm_hid_size, dim=-1), dim=-2)
        """
            entity preds
        """ 
        ent_logits = self.ent_output(text_fusion_token) # bs*len*ent_label_num
        ent_logits = self.ent_dropout(ent_logits)

        img_logits = self.classification_img(img_fusion)
        class_num = img_logits.shape[-1]
        pred_boxes = self.bbox_head(img_fusion).sigmoid()

        if is_train:
            ent_loss = self.calu_ent_loss(grid_labels_ent, ent_logits)

            ent_link_logits = self.get_ent_pair_tensor(ent_link, ent_link_ids.shape[1], task="link")

            ent_rel_logits = self.get_ent_pair_tensor(ent_rel, ent_rel_ids.shape[1], task="rel")

            if not ent_link_ids.shape[-1]:
                ent_link_ids = torch.zeros((ent_link_logits.shape[0], ent_link_logits.shape[1])).to(self.config.device).long()
            link_loss = self.loss_fct(ent_link_logits.view(-1, 2), ent_link_ids.view(-1))

            if not ent_rel_ids.shape[-1]:
                ent_rel_ids = torch.zeros((ent_rel_logits.shape[0], ent_rel_logits.shape[1])).to(self.config.device).long()
            rel_loss = self.loss_fct(ent_rel_logits.view(-1, self.config.rel_label_num), ent_rel_ids.view(-1))
            
            img_loss = self.calu_img_loss(img_logits.view(self.bs, -1, class_num).permute(0, 2, 1), pred_boxes, grid_labels_img[:,:,:,:1].view(self.bs, -1).long(), grid_labels_img[:,:,:,1:])
            
            return ent_loss + link_loss + rel_loss + img_loss
        else:
            ent_p, ent_r, ent_c, rel_p, rel_c, rel_r = 0, 0, 0, 0, 0, 0
            rel_p_gold, rel_c_gold, rel_r_gold = 0, 0, 0
            batch_ceaf_c, batch_ceaf_p, batch_ceaf_r, batch_muc_c, batch_muc_p, batch_muc_r, batch_b3_c_p, batch_b3_c_r, batch_b3_p, batch_b3_r = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            link_rel_p, link_rel_c, link_rel_r = 0, 0, 0
            link_rel_p_gold, link_rel_c_gold, link_rel_r_gold = 0, 0, 0
            batch_ceaf_c_gold, batch_ceaf_p_gold, batch_ceaf_r_gold, batch_muc_c_gold, batch_muc_p_gold, batch_muc_r_gold, batch_b3_c_p_gold, batch_b3_c_r_gold, batch_b3_p_gold, batch_b3_r_gold = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            batch_ground_p, batch_ground_r, batch_ground_c = 0, 0, 0
            error_ent_be, error_ent_te, error_link_p, error_link_ee, error_link_se, error_rel_ne, error_rel_te, error_gro_be, error_gro_te = 0, 0, 0, 0, 0, 0, 0, 0, 0

            """entity evluate"""
            outputs = self.crf.decode(ent_logits)
            ent_p, ent_r, ent_c, error_ent_be, error_ent_te, all_link, all_ent_pair_base_link_list, all_link_base_gold_ent, all_ent_pair_base_gold_link_list = self.ent_decode(outputs, ent_outputs, link_outputs, grid_labels_ent)

            """link evluate base gold entity"""
            batch_ceaf_c_gold, batch_ceaf_p_gold, batch_ceaf_r_gold, batch_muc_c_gold, batch_muc_p_gold, batch_muc_r_gold, batch_b3_c_p_gold, batch_b3_c_r_gold, batch_b3_p_gold, batch_b3_r_gold, error_link_p, error_link_se, error_link_ee = self.ent_link_decode(ent_pair_rel, link_outputs)

            """entity relation evluate base gold link"""
            rel_p_gold, rel_c_gold, rel_r_gold, all_link_pair_type_gold = self.ent_rel_decode(all_ent_pair_base_gold_link_list, ent_rel)

            """link relation evluate base gold link"""
            link_rel_p_gold, link_rel_c_gold, link_rel_r_gold, error_rel_te, error_rel_ne = self.link_rel_preds(all_link_pair_type_gold, rel_outputs)

            batch_ground_p, batch_ground_r, batch_ground_c, error_gro_be, error_gro_te = self.ground_encode(torch.argmax(img_logits, -1), pred_boxes, grid_labels_img)

            return ent_p, ent_r, ent_c, error_ent_be, error_ent_te, rel_p, rel_c, rel_r, batch_ceaf_c, batch_ceaf_p, batch_ceaf_r, batch_muc_c, batch_muc_p, batch_muc_r, batch_b3_c_p, batch_b3_c_r, batch_b3_p, batch_b3_r, error_link_p, error_link_se, error_link_ee, link_rel_p, link_rel_c, link_rel_r, error_rel_te, error_rel_ne, batch_ceaf_c_gold, batch_ceaf_p_gold, batch_ceaf_r_gold, batch_muc_c_gold, batch_muc_p_gold, batch_muc_r_gold, batch_b3_c_p_gold, batch_b3_c_r_gold, batch_b3_p_gold, batch_b3_r_gold, rel_p_gold, rel_c_gold, rel_r_gold, link_rel_p_gold, link_rel_c_gold, link_rel_r_gold, batch_ground_p, batch_ground_r, batch_ground_c, error_gro_be, error_gro_te