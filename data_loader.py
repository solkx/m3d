import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import prettytable as pt
from gensim.models import KeyedVectors
from transformers import AutoTokenizer
import os
import utils
import requests
import cv2
from tqdm import tqdm
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"

dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class Vocabulary(object):
    PAD = '<pad>'
    UNK = '<unk>'
    SUC = '<suc>'
    PRE = '<pre>'
    NSUC = '<nsuc>'
    DSUC = '<dsuc>'
    CON = '<con>'



    def __init__(self, frequency=0):
        self.token2id = {self.PAD: 0, self.UNK: 1}
        self.id2token = {0: self.PAD, 1: self.UNK}
        self.token2count = {self.PAD: 1000, self.UNK: 1000}
        self.frequency = frequency

        self.char2id = {self.PAD: 0, self.UNK: 1}
        self.id2char = {0: self.PAD, 1: self.UNK}

        self.label2id_ent = {"B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-TIME": 5, "I-TIME":6, "B-ORG": 7, "I-ORG": 8, "O": 0}
        self.label2id_rel = {"None": 0}

        self.id2label_ent = {1: "B-PER", 2: "I-PER", 3: "B-LOC", 4: "I-LOC", 5: "B-TIME", 6: "I-TIME", 7: "B-ORG", 8: "I-ORG", 0: "O"}
        self.id2label_rel = {0: "None"}
        self.label2id_gro = {self.PAD: 0, "PER": 1, "LOC": 2, "ORG": 3}
        self.id2label_gro = {0: self.PAD, 1: "PER", 2: "LOC", 3: "ORG"}

    def add_token(self, token):
        token = token.lower()
        if token in self.token2id:
            self.token2count[token] += 1
        else:
            self.token2id[token] = len(self.token2id)
            self.id2token[self.token2id[token]] = token
            self.token2count[token] = 1

        assert token == self.id2token[self.token2id[token]]

    def add_char(self, char):
        char = char.lower()
        if char not in self.char2id:
            self.char2id[char] = len(self.char2id)
            self.id2char[self.char2id[char]] = char

        assert char == self.id2char[self.char2id[char]]

    def add_label(self, label):
        label = label.lower()
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label

        assert label == self.id2label[self.label2id[label]]

    def remove_low_frequency_token(self):
        new_token2id = {self.PAD: 0, self.UNK: 1}
        new_id2token = {0: self.PAD, 1: self.UNK}

        for token in self.token2id:
            if self.token2count[token] > self.frequency and token not in new_token2id:
                new_token2id[token] = len(new_token2id)
                new_id2token[new_token2id[token]] = token

        self.token2id = new_token2id
        self.id2token = new_id2token

    def __len__(self):
        return len(self.token2id)

    def encode(self, text):
        return [self.token2id.get(x.lower(), 1) for x in text]

    def encode_char(self, text):
        return [self.char2id.get(x, 1) for x in text]

    def decode(self, ids):
        return [self.id2token.get(x) for x in ids]


def pad_and_stack_tensors(tensor_group):
    # 找到这一组张量的 x_max
    x_max = max(tensor.shape[0] for tensor in tensor_group)
    
    # 创建一个列表，用于存储填充后的张量
    padded_tensors = []
    
    for tensor in tensor_group:
        x = tensor.shape[0]
        
        # 如果当前张量的 x 小于 x_max，进行填充
        if x < x_max:
            # 使用torch.nn.functional.pad在第0维进行填充
            padding = (0, 0, 0, 0, 0, 0, 0, x_max - x)  # 只在第0维填充 (x_max - x)
            padded_tensor = torch.nn.functional.pad(tensor, padding, mode='constant', value=0)
        else:
            padded_tensor = tensor  # 如果已经是 x_max 的大小，无需填充
        
        padded_tensors.append(padded_tensor)
    
    # 按第一个维度进行堆叠，形成 (8, x_max, 3, 1920, 1080) 的张量
    stacked_tensor = torch.stack(padded_tensors)
    
    return stacked_tensor

def pad_ground_tensors(tensor_group):
    # 找到这一组张量的 x_max
    x_max = max(tensor.shape[0] for tensor in tensor_group)
    
    # 创建一个列表，用于存储填充后的张量
    padded_tensors = []
    
    for tensor in tensor_group:
        x, m, n = tensor.shape
        
        # 如果当前张量的 x 小于 x_max，进行填充
        if x < x_max:
            # 使用torch.nn.functional.pad在第0维进行填充
            padded_tensor = torch.cat([tensor, torch.zeros(((x_max-x), m, n)).float()], dim=0)
        
            padded_tensors.append(padded_tensor)
        else:
            padded_tensors.append(tensor)
    
    # 按第一个维度进行堆叠，形成 (8, x_max, 3, 1920, 1080) 的张量
    stacked_tensor = torch.stack(padded_tensors)
    
    return stacked_tensor

def collate_fn(data):
    bert_inputs, bert_inputs_token, bert_inputs_mask, ent_output_ids, grid_labels_ent, ent_output_mask, ent_link_ids, ent_link_mask, ent_rel_ids, ent_rel_mask, images, grid_labels_img, pieces2word, ent_outputs, link_outputs, rel_outputs, docs, ent_link, ent_rel, ent_pair_rel, sent_length, dataType = map(list, zip(*data))

    batch_size = len(bert_inputs)
    max_tok = np.max(sent_length)
    sent_length = torch.LongTensor(sent_length)
    max_pie = np.max([x.shape[0] for x in bert_inputs_token])
    grid_labels_ent = pad_sequence(grid_labels_ent, True, padding_value=-100)

    bert_inputs = pad_sequence(bert_inputs, True)
    bert_inputs_token = pad_sequence(bert_inputs_token, True)
    bert_inputs_mask = pad_sequence(bert_inputs_mask, True)
    ent_output_ids = pad_sequence(ent_output_ids, True)
    ent_output_mask = pad_sequence(ent_output_mask, True)
    ent_link_ids = pad_sequence(ent_link_ids, True)
    ent_link_mask = pad_sequence(ent_link_mask, True)
    ent_rel_ids = pad_sequence(ent_rel_ids, True)
    ent_rel_mask = pad_sequence(ent_rel_mask, True)
    images = pad_and_stack_tensors(images)
    grid_labels_img = pad_ground_tensors(grid_labels_img)   

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data
    
    sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
    pieces2word = fill(pieces2word, sub_mat)

    return bert_inputs, bert_inputs_token, bert_inputs_mask, ent_output_ids, grid_labels_ent, ent_output_mask, ent_link_ids, ent_link_mask, ent_rel_ids, ent_rel_mask, images, grid_labels_img, pieces2word, ent_outputs, link_outputs, rel_outputs, docs, ent_link, ent_rel, ent_pair_rel, sent_length, dataType


class RelationDataset(Dataset):
    def __init__(self, bert_inputs, bert_inputs_token, bert_inputs_mask, ent_output_ids, grid_labels_ent, ent_output_mask, ent_link_ids, ent_link_mask, ent_rel_ids, ent_rel_mask, images, grid_labels_img, pieces2word, ent_outputs, link_outputs, rel_outputs, docs, ent_link, ent_rel, ent_pair_rel, sent_legth, dataType):
        self.bert_inputs = bert_inputs
        self.bert_inputs_token = bert_inputs_token
        self.ent_outputs = ent_outputs
        self.ent_output_ids = ent_output_ids
        self.grid_labels_ent = grid_labels_ent
        self.link_outputs = link_outputs
        self.rel_outputs = rel_outputs
        self.ent_output_mask = ent_output_mask
        self.bert_inputs_mask = bert_inputs_mask
        self.images = images
        self.grid_labels_img = grid_labels_img
        self.docs = docs
        self.ent_link_ids = ent_link_ids
        self.ent_link_mask = ent_link_mask
        self.ent_rel_ids = ent_rel_ids
        self.ent_rel_mask = ent_rel_mask
        self.ent_link = ent_link
        self.ent_rel = ent_rel
        self.ent_pair_rel = ent_pair_rel
        self.pieces2word = pieces2word
        self.sent_length = sent_legth
        self.dataType = dataType


    def __getitem__(self, item):
        return torch.LongTensor(self.bert_inputs[item]), \
               torch.LongTensor(self.bert_inputs_token[item]), \
               torch.LongTensor(self.bert_inputs_mask[item]), \
               torch.LongTensor(self.ent_output_ids[item]), \
               torch.LongTensor(self.grid_labels_ent[item]), \
               torch.LongTensor(self.ent_output_mask[item]), \
               torch.LongTensor(self.ent_link_ids[item]), \
               torch.LongTensor(self.ent_link_mask[item]), \
               torch.LongTensor(self.ent_rel_ids[item]), \
               torch.LongTensor(self.ent_rel_mask[item]), \
               torch.LongTensor(self.images[item]), \
               torch.FloatTensor(self.grid_labels_img[item]), \
               torch.LongTensor(self.pieces2word[item]), \
               self.ent_outputs[item], \
               self.link_outputs[item], \
               self.rel_outputs[item], \
               self.docs[item], \
               self.ent_link[item], \
               self.ent_rel[item], \
               self.ent_pair_rel[item], \
               self.sent_length[item], \
               self.dataType[item]
               

    def __len__(self):
        return len(self.bert_inputs)

def dataFormat(instance):
    link_id2index = {}
    for linkID, linkItem in instance["entityLink"].items():
        index_list = []
        for link in linkItem["link"]:
            index_list.append((link["start"], link["end"]-1, link["type"]))
        index_list = sorted(index_list, key=lambda x: x[0])
        link_id2index[linkID] = sorted(index_list, key=lambda x: x[0])
    return link_id2index


def process_bert(data, tokenizer, vocab, config):
    bert_inputs = []
    bert_inputs_token = []
    ent_outputs = []
    ent_output_ids = []
    link_outputs = []
    grid_labels_ent = []
    rel_outputs = []
    ent_output_mask = []
    bert_inputs_mask = []
    images = []
    grid_labels_img = []
    docs = []
    ent_link = []
    ent_link_ids = []
    ent_link_mask = []
    ent_rel = []
    ent_rel_ids = []
    ent_rel_mask = []
    ent_pair_rel = []
    images = []
    pieces2word = []
    sent_length = []
    dataType = []
    max_len, max_imgNum = 0, 0
    for instance in tqdm(data):
        doc_lang = instance["lang"]
        if doc_lang == "zh":
            docList = list(instance['doc'])
        else:
            docList = instance["doc"].split(" ")
        if instance["doc"] and instance["video_id"]:
            dataType.append(0)
        elif not instance["doc"] and instance["video_id"]:
            dataType.append(1)
        elif instance["doc"] and not instance["video_id"]:
            dataType.append(2)

        if len(docList) == 0:
            docList = [tokenizer.pad_token]

        tokens_token = [tokenizer.tokenize(word) for word in docList]
        pieces_token = [piece for pieces in tokens_token for piece in pieces]
        _bert_inputs_token = tokenizer.convert_tokens_to_ids(pieces_token)
        _bert_inputs_token = np.array([tokenizer.cls_token_id] + _bert_inputs_token + [tokenizer.sep_token_id])

        length = len(docList)

        _pieces2word = np.zeros((length, len(_bert_inputs_token)), dtype=np.bool_)

        if tokenizer is not None:
            start = 0
            for i, pieces in enumerate(tokens_token):
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                start += len(pieces)

        _grid_labels_ent = ['O'] * length
        link_id2index = dataFormat(instance)
        _link_text = {}
        _ent_text = []
        temp = []
        for link_id, linkItem in link_id2index.items():
            # preId = -1
            for ent in linkItem:
                temp.append(f"{ent[0]}-{ent[1]}##{vocab.label2id_ent[f'B-{ent[-1]}']}")
                _grid_labels_ent[ent[0]] = f'B-{ent[-1]}'  # 实体的开始
                for idx in range(ent[0] + 1, ent[1] + 1):
                    _grid_labels_ent[idx] = f'I-{ent[-1]}'  # 实体的内部

                _ent_text.append(f"{ent[0]}-{ent[1]}##{vocab.label2id_ent[f'B-{ent[-1]}']}")
            _link_text[link_id] = temp
            temp = []
        _grid_labels_ent = [vocab.label2id_ent[label] for label in _grid_labels_ent]

        tokens = [["[CLS]"]] + [[word] for word in docList] + [["[SEP]"]]
        pieces = [piece for pieces in tokens for piece in pieces]

        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        
        _ent_output_mask, _bert_inputs_mask = [1] * len(_bert_inputs), [1] * len(_bert_inputs)
        _ent_outputs, _ent_output_ids, _link_outputs, _rel_outputs, _ent_link, _ent_link_ids, _ent_rel, _ent_rel_ids, _ent_pair_rel = utils.resultFormat(instance, vocab, tokens, pieces)


        video_id = instance['video_id']
        _image = []
        if video_id:
            rootPath = f"./img/{video_id}"
            _grid_labels_img = []
            for imgPath in os.listdir(rootPath):
                if "txt" not in imgPath:
                    image = cv2.resize(cv2.imread(f'{rootPath}/{imgPath}'), (config.height, config.width), interpolation=cv2.INTER_LANCZOS4)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_tensor = torch.tensor(image).permute(2, 0, 1).float() / 255.0
                    _image.append(image_tensor)
                else:
                    _grid_labels_temp = np.zeros((config.maxGroundNum, 5), dtype=np.float)
                    with open(f"{rootPath}/{imgPath}", "r", encoding="utf-8") as f:
                        for i, line in enumerate(f.read().split("\n")):
                            item_list = line.split(" ")
                            _grid_labels_temp[i, 0] = vocab.label2id_gro[item_list[0]] 
                            for j, item in enumerate(item_list[1:]):
                                _grid_labels_temp[i, j+1] = float(item)
                    _grid_labels_img.append(_grid_labels_temp)
            _grid_labels_img = np.array(_grid_labels_img)         

        if _image:
            _image = torch.stack(_image, dim=0)
        else:
            _image = torch.zeros((1, 3, config.height, config.width))
            _grid_labels_img = np.zeros((1, config.maxGroundNum, 5), dtype=np.float)
        _image = _image.long()

        if _image.shape[0] > max_imgNum:
            max_imgNum = _image.shape[0]

        if _bert_inputs.__len__() > max_len:
            max_len = _bert_inputs.__len__()

        _ent_link_mask = [1] * len(_ent_link_ids)
        _ent_rel_mask = [1] * len(_ent_rel_ids)

        sent_length.append(length)      
        bert_inputs.append(_bert_inputs)
        bert_inputs_token.append(_bert_inputs_token)
        pieces2word.append(_pieces2word)
        ent_output_ids.append(_ent_output_ids)
        grid_labels_ent.append(_grid_labels_ent)
        bert_inputs_mask.append(_bert_inputs_mask)
        images.append(_image)
        grid_labels_img.append(_grid_labels_img)
        link_outputs.append(_link_outputs)
        rel_outputs.append(_rel_outputs)
        ent_outputs.append(_ent_outputs)
        ent_output_mask.append(_ent_output_mask)
        docs.append(pieces)
        ent_link.append(_ent_link)
        ent_link_ids.append(_ent_link_ids)
        ent_link_mask.append(_ent_link_mask)
        ent_rel.append(_ent_rel)
        ent_rel_ids.append(_ent_rel_ids)
        ent_rel_mask.append(_ent_rel_mask)
        ent_pair_rel.append(_ent_pair_rel)

        

    return bert_inputs, bert_inputs_token, bert_inputs_mask, ent_output_ids, grid_labels_ent, ent_output_mask, ent_link_ids, ent_link_mask, ent_rel_ids, ent_rel_mask, images, grid_labels_img, pieces2word, ent_outputs, link_outputs, rel_outputs, docs, ent_link, ent_rel, ent_pair_rel, sent_length, dataType


def fill_vocab(vocab, dataset):
    entity_num, link_num, rel_num, img_num, ground_num, ground_num_max = 0, 0, 0, 0, 0, 0
    for instance in dataset:
        video_id = instance['video_id']
        img_root = "./img"
        if img_num < len(os.listdir(f"{img_root}/{video_id}")) // 2:
            img_num = len(os.listdir(f"{img_root}/{video_id}")) // 2
        for filePath in os.listdir(f"{img_root}/{video_id}"):
            if ".txt" in filePath:
                with open(f"{img_root}/{video_id}/{filePath}", "r", encoding="utf-8") as f:
                    b_num = len(f.read().split("\n"))
                    ground_num += b_num
                    if ground_num_max < b_num:
                        ground_num_max = b_num
        link_num += len(instance["entityLink"])
        for link in instance["entityLink"].values():
            entity_num += link["len"]
        rel_num += len(instance["relation"])
        for rel in instance["relation"]:
            if rel["type"] not in vocab.label2id_rel:
                vocab.label2id_rel[rel["type"]] = len(vocab.label2id_rel)
                vocab.id2label_rel[len(vocab.id2label_rel)] = rel["type"]
    return entity_num, link_num, rel_num, img_num, ground_num, ground_num_max

def miss_pro(data):
    random.shuffle(data)
    l = len(data)
    total = data[:l//3]
    miss_text = data[l//3:l//3*2]
    miss_video = data[l//3*2:]
    new_miss_text, new_miss_video = [], []
    for m_text in miss_text:
        new_miss_text.append({
            "doc": "",
            "relation": [],
            "entityLink": {},
            "image": m_text["image"],
            "video_id": m_text["video_id"],
            "lang": m_text["lang"]
        })
    for m_video in miss_video:
        new_miss_video.append({
            "doc": m_video["doc"],
            "relation": m_video["relation"],
            "entityLink": m_video["entityLink"],
            "image": [],
            "video_id": "",
            "lang": m_video["lang"]
        })
    total.extend(new_miss_text)
    total.extend(new_miss_video)
    random.shuffle(total)
    return total

def load_data_bert(config):
    with open(f'./data/{config.lang}/train_{config.lang}.json', 'r', encoding='utf-8') as f:
        train_data = miss_pro(json.loads(f.read()))
    with open(f'./data/{config.lang}/dev_{config.lang}.json', 'r', encoding='utf-8') as f:
        dev_data = miss_pro(json.loads(f.read()))
    with open(f'./data/{config.lang}/test_{config.lang}.json', 'r', encoding='utf-8') as f:
        test_data = miss_pro(json.loads(f.read()))


    tokenizer = None
    while tokenizer is None:
        try:
            # tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir="./cache/")
            tokenizer = AutoTokenizer.from_pretrained(
                config.bert_name,
                max_length=config.max_sequence_length,
                padding="max_length",
                truncation=True,
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            continue


    vocab = Vocabulary()
    train_ent_num, trtain_link_num, train_rel_num, train_img_num, train_ground_num, train_ground_num_max = fill_vocab(vocab, train_data)
    dev_ent_num, dev_link_num, dev_rel_num, dev_img_num, dev_ground_num, dev_ground_num_max = fill_vocab(vocab, dev_data)
    test_ent_num, test_link_num, test_rel_num, test_img_num, test_ground_num, test_ground_num_max = fill_vocab(vocab, test_data)

    table = pt.PrettyTable(["", 'docs', 'entities', "links", "relation", "grounding"])
    table.add_row(['train', len(train_data), train_ent_num, trtain_link_num, train_rel_num, train_ground_num])
    table.add_row(['dev', len(dev_data), dev_ent_num, dev_link_num, dev_rel_num, dev_ground_num])
    table.add_row(['test', len(test_data), test_ent_num, test_link_num, test_rel_num, test_ground_num])
    config.logger.info("\n{}".format(table))

    config.word_num = len(vocab.token2id)
    config.char_num = len(vocab.char2id)
    config.ent_label_num = len(vocab.label2id_ent)
    config.rel_label_num = len(vocab.label2id_rel)
    config.vocab = vocab
    config.maxImgNum = max([train_img_num, dev_img_num, test_img_num])
    config.maxGroundNum = max([train_ground_num_max, dev_ground_num_max, test_ground_num_max])

    train_dataset = RelationDataset(*process_bert(train_data, tokenizer, vocab, config))
    dev_dataset = RelationDataset(*process_bert(dev_data, tokenizer, vocab, config))
    test_dataset = RelationDataset(*process_bert(test_data, tokenizer, vocab, config))
    return train_dataset, dev_dataset, test_dataset


def load_embedding(config):
    vocab = config.vocab
    wvmodel = KeyedVectors.load_word2vec_format(config.embedding_path, binary=True)
    embed_size = config.word_emb_size
    embedding = np.random.uniform(-0.01, 0.01, (len(vocab), embed_size))
    hit = 0
    for token, i in vocab.token2id.items():
        if token in wvmodel:
            hit += 1
            embedding[i, :] = wvmodel[token]
    print("Total hit: {} rate {:.4f}".format(hit, hit / len(vocab)))
    embedding[0] = np.zeros(embed_size)
    embedding = torch.FloatTensor(embedding)
    return embedding
