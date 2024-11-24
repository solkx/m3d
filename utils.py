import logging
import time
import pickle
from scipy.optimize import linear_sum_assignment
import numpy as np
import itertools
import os

def get_logger(config):
    if not os.path.exists(config.logPath):
        os.makedirs(config.logPath)
    pathname = f'./{config.logPath}/{config.lang}_{config.seed}_{config.epochs}_{config.batch_size}_{config.learning_rate}_{config.bert_learning_rate}_{time.strftime("%m-%d_%H-%M-%S")}.txt'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def save_file(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def get_pieces_index(tokens):
    pieces2tokens = []
    for ind, pieces in enumerate(tokens):
        for _ in pieces:
            pieces2tokens.append(ind)
    return pieces2tokens

"""
    1. Obtain all entity chain pairs.
    2. mentions is list, type of every item is String, example mentions=["ent1 start index:ent1 end index_ent2 start index:ent2 end index_...", ...]
"""
def get_pair(list1, list2=None):
    res = []
    if list2:
        for i, m1 in enumerate(list1):
            for j, m2 in enumerate(list2):
                # if i <= j:
                res.append(f"{m1}-{m2}")
                res.append(f"{m2}-{m1}")
    else:
        for i, m1 in enumerate(list1):
            for j, m2 in enumerate(list1):
                if i < j:
                    res.append(f"{m1}-{m2}")
    return res

def merge_overlapping_sublists(lists):
    if len(lists) == 1:
        lists.append([lists[0][0]])
    merged_lists = []
    
    # 遍历每个子列表
    for sublist in lists:
        # 将当前子列表转换为集合
        current_set = set(sublist)
        
        # 检查是否存在与当前子列表重叠的集合
        merged = False
        for i, merged_list in enumerate(merged_lists):
            if current_set & set(merged_list):  # 有交集
                merged_lists[i].extend(sublist)  # 合并子列表
                merged_lists[i] = list(set(merged_lists[i]))  # 去重
                merged = True
                break
        
        # 如果没有与现有合并列表重叠，添加一个新列表
        if not merged:
            merged_lists.append(sublist)
    
    return merged_lists
 
def resultFormat(item, vocab, tokens, pieces):
    pieces2token = get_pieces_index(tokens)
    BIO_list = ["O"] * len(pieces2token)
    golds = []
    linkList = item["entityLink"]
    link_id_dic = {} # {Co referential chain 1_id:"Co referential chain 1 entity 1 index_Co referential chain 1 entity 2 index_..._Co referential chain 1 entity n index", ...}
    for link_id, link in linkList.items():
        temp = []
        
        entity_type = link["type"]
        for entityItem in link["link"]:
            tokes_s = entityItem["start"] + 1
            tokes_e = entityItem["end"] + 1
            s = pieces2token.index(tokes_s)
            e = pieces2token.index(tokes_e)
            temp.append(f"{s}:{e}:{entity_type}:{''.join(pieces[s:e]).replace('-', '——')}")
            golds.append(f"{s}:{e}:{entity_type}:{''.join(pieces[s:e]).replace('-', '——')}")
            BIO_list[s] = f"B-{entity_type}"
            for i in range(s+1, e):
                BIO_list[i] = f"I-{entity_type}"
        link_id_dic[link_id] = temp
    doc_all_relation = item["relation"]
    rel_list = {"num":len(doc_all_relation)} # [relation type1:[link1, link2, ...], ...] 
    ent_rel_not_none_dic = {}
    for rel in doc_all_relation:
        link1 = list(link_id_dic[rel["link1"]])
        link2 = list(link_id_dic[rel["link2"]])
        link_ent_pair = set(get_pair(link1, link2))
        for pair in link_ent_pair:
            ent_rel_not_none_dic[pair] = rel['type']
        if rel["type"] not in rel_list:
            rel_list[rel["type"]] = [link_ent_pair]
        else:
            rel_list[rel["type"]].append(link_ent_pair)
    bio_id_list = [vocab.label2id_ent[bio] for bio in BIO_list]
    link_list = [v for v in link_id_dic.values()] # [[Co referential chain 1 entity 1 index, ..., Co referential chain 1 entity n index], ..., [Co referential chain N entity 1 index, ... Co referential chain N entity n index]]
    ent_link_not_none_dic = []
    for link in link_list:
        ent_link_not_none_dic.extend(get_pair(link))
    ent_pair_rel = get_pair(golds)
    ent_link, ent_link_ids = get_all_ent_pair_link_list(ent_link_not_none_dic, ent_pair_rel, vocab)
    ent_rel, ent_rel_ids = get_all_ent_pair_rel_list(ent_rel_not_none_dic, ent_pair_rel, vocab)
    return golds, bio_id_list, link_list, rel_list, ent_link, ent_link_ids, ent_rel, ent_rel_ids, ent_pair_rel

def get_all_ent_pair_link_list(ent_rel_not_none_dic, ent_pair_rel, vocab):
    ent_rel, ent_rel_ids = [], []
    rel2ids = vocab.label2id_rel
    for ent_pair in ent_pair_rel:
        if ent_pair in ent_rel_not_none_dic:
            ent_rel.append(f"{ent_pair}:conf")
            ent_rel_ids.append(1)
        else:
            ent_rel.append(f"{ent_pair}:None") 
            ent_rel_ids.append(0)
    return ent_rel, ent_rel_ids


def get_all_ent_pair_rel_list(ent_rel_not_none_dic, ent_pair_rel, vocab):
    ent_rel, ent_rel_ids = [], []
    rel2ids = vocab.label2id_rel
    for ent_pair in ent_pair_rel:
        if ent_pair in ent_rel_not_none_dic:
            ent_rel.append(f"{ent_pair}:{ent_rel_not_none_dic[ent_pair]}")
            ent_rel_ids.append(rel2ids[ent_rel_not_none_dic[ent_pair]])
        else:
            ent_rel.append(f"{ent_pair}:None") 
            ent_rel_ids.append(rel2ids["None"])
    return ent_rel, ent_rel_ids

def muc(predicted_clusters, gold_clusters):      
    """
        predicted_clusters      list(list)       预测实体簇
        gold_clusters           list(list)       标注实体簇
    """  
    pred_edges = set()
    for cluster in predicted_clusters:
        pred_edges |= set(itertools.combinations(cluster, 2))
    gold_edges = set()
    for cluster in gold_clusters:
        gold_edges |= set(itertools.combinations(cluster, 2))
    correct_edges = gold_edges & pred_edges
    return len(correct_edges), len(pred_edges), len(gold_edges)


def b3(predicted_clusters, gold_clusters):
    """
        predicted_clusters      list(list)       预测实体簇
        gold_clusters           list(list)       标注实体簇
    """  
    mentions = set(sum(predicted_clusters, [])) & set(sum(gold_clusters, []))
    precisions = []
    recalls = []
    for mention in mentions:
        mention2predicted_cluster = [x for x in predicted_clusters if mention in x][0]
        mention2gold_cluster = [x for x in gold_clusters if mention in x][0]
        corrects = set(mention2predicted_cluster) & set(mention2gold_cluster)
        precisions.append(len(corrects) / len(mention2predicted_cluster))
        recalls.append(len(corrects) / len(mention2gold_cluster))
    return sum(precisions), sum(recalls), len(precisions), len(recalls)


def ceaf(predicted_clusters, gold_clusters):
    """
        predicted_clusters      list(list)       预测实体簇
        gold_clusters           list(list)       标注实体簇
    """  
    scores = np.zeros((len(predicted_clusters), len(gold_clusters)))
    for j in range(len(gold_clusters)):
        for i in range(len(predicted_clusters)):
            scores[i, j] = len(set(predicted_clusters[i]) & set(gold_clusters[j]))
    indexs = linear_sum_assignment(scores, maximize=True)
    max_correct_mentions = sum(
        [scores[indexs[0][i], indexs[1][i]] for i in range(indexs[0].shape[0])]
    )
    return max_correct_mentions, len(sum(predicted_clusters, [])), len(sum(gold_clusters, []))