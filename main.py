import argparse
import random
import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import torch.optim as optim
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
import data_loader
import utils
from model import Model
import os
import torch.nn.functional as F

def error(a, b, c, a_text, b_text):
    if c:
        a_r = a / c
        b_r = b / c
    else:
        a_r, b_r = 0, 0
    return f"{round(a_r, 4)}({a_text})", f"{round(b_r, 4)}({b_text})"

class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        bert_params = set(self.model.bert.parameters())
        vit_params = set(self.model.vit.parameters())
        other_params = list(set(self.model.parameters()) - bert_params - vit_params)
        # other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': config.bert_learning_rate,
                'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
                'lr': config.bert_learning_rate,
                'weight_decay': 0.0},
            {'params': [p for n, p in model.vit.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': config.weight_decay},
            {'params': other_params,
                'lr': config.learning_rate,
                'weight_decay': config.weight_decay},
        ]

        self.optimizer = optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps=config.warm_factor * updates_total,
                                                                      num_training_steps=updates_total)

    def calu_res(self, p, r, c_p, c_r=None):
        if not c_r:
            c_r = c_p
        try:
            precious = c_p / p
        except ZeroDivisionError:
            precious = 0
        try:
            recall = c_r / r
        except ZeroDivisionError:
            recall = 0

        try:
            f1 = 2 * precious * recall / (precious + recall)
        except ZeroDivisionError:
            f1 = 0
        return round(precious, 4), round(recall, 4), round(f1, 4)
        
    def train(self, epoch, data_loader):
        self.model.train()
        loss_list = []
        for i, data_batch in enumerate(tqdm(data_loader)):
            ent_outputs, link_outputs, rel_outputs, docs, ent_link, ent_rel, ent_pair_rel, sent_length, dataType = data_batch[-9:]

            data_batch = [data.to(device) for data in data_batch[:-9]]

            bert_inputs, bert_inputs_token, bert_inputs_mask, ent_output_ids, grid_labels_ent, ent_output_mask, ent_link_ids, ent_link_mask, ent_rel_ids, ent_rel_mask, images, grid_labels_img, pieces2word = data_batch
            
            loss = model(bert_inputs, bert_inputs_token, bert_inputs_mask, ent_output_ids, grid_labels_ent, ent_output_mask, ent_link_ids, ent_link_mask, ent_rel_ids, ent_rel_mask, images, grid_labels_img, pieces2word, ent_outputs, link_outputs, rel_outputs, docs, ent_link, ent_rel, ent_pair_rel, dataType)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_list.append(loss.cpu().item())

            self.scheduler.step()

        table = pt.PrettyTable(["Train {}".format(epoch), "Loss"])
        table.add_row(["", "{:.4f}".format(np.mean(loss_list))])
        logger.info("\n{}".format(table))

        return loss

    def eval(self, epoch, data_loader, is_test=False):
        self.model.eval()
        total_p, total_r, total_c = 0, 0, 0
        total_rel_p_gold, total_rel_r_gold, total_rel_c_gold = 0, 0, 0
        total_ent_rel_p_gold, total_ent_rel_r_gold, total_ent_rel_c_gold = 0, 0, 0
        total_ceaf_c_gold, total_ceaf_p_gold, total_ceaf_r_gold, total_muc_c_gold, total_muc_p_gold, total_muc_r_gold, total_b3_c_p_gold, total_b3_c_r_gold, total_b3_p_gold, total_b3_r_gold = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        total_ground_c, total_ground_p, total_ground_r,  = 0, 0, 0
        total_error_ent_be, total_error_ent_te, total_error_link_p, total_error_link_ee, total_error_link_se, total_error_rel_te, total_error_rel_ne, total_error_gro_be, total_error_gro_te = 0, 0, 0, 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for i, data_batch in enumerate(tqdm(data_loader)):
                ent_outputs, link_outputs, rel_outputs, docs, ent_link, ent_rel, ent_pair_rel, sent_length, dataType = data_batch[-9:]
                data_batch = [data.to(device) for data in data_batch[:-9]]

                bert_inputs, bert_inputs_token, bert_inputs_mask, ent_output_ids, grid_labels_ent, ent_output_mask, ent_link_ids, ent_link_mask, ent_rel_ids, ent_rel_mask, images, grid_labels_img, pieces2word = data_batch
                
                ent_p, ent_r, ent_c, error_ent_be, error_ent_te, rel_p, rel_c, rel_r, ceaf_c, ceaf_p, ceaf_r, muc_c, muc_p, muc_r, b3_c_p, b3_c_r, b3_p, b3_r, error_link_p, error_link_se, error_link_ee, link_rel_p, link_rel_c, link_rel_r, error_rel_te, error_rel_ne, ceaf_c_gold, ceaf_p_gold, ceaf_r_gold, muc_c_gold, muc_p_gold, muc_r_gold, b3_c_p_gold, b3_c_r_gold, b3_p_gold, b3_r_gold, rel_p_gold, rel_c_gold, rel_r_gold, link_rel_p_gold, link_rel_c_gold, link_rel_r_gold, batch_ground_p, batch_ground_r, batch_ground_c, error_gro_be, error_gro_te = model(bert_inputs, bert_inputs_token, bert_inputs_mask, ent_output_ids, grid_labels_ent, ent_output_mask, ent_link_ids, ent_link_mask, ent_rel_ids, ent_rel_mask, images, grid_labels_img, pieces2word, ent_outputs, link_outputs, rel_outputs, docs, ent_link, ent_rel, ent_pair_rel, dataType, is_train=False)

                total_p += ent_p 
                total_r += ent_r
                total_c += ent_c
                total_error_ent_be += error_ent_be
                total_error_ent_te += error_ent_te

                total_ent_rel_p_gold += rel_p_gold
                total_ent_rel_r_gold += rel_r_gold
                total_ent_rel_c_gold += rel_c_gold

                total_ceaf_c_gold += ceaf_c_gold
                total_ceaf_p_gold += ceaf_p_gold
                total_ceaf_r_gold += ceaf_r_gold
                total_muc_c_gold += muc_c_gold
                total_muc_p_gold += muc_p_gold
                total_muc_r_gold += muc_r_gold
                total_b3_c_p_gold += b3_c_p_gold
                total_b3_c_r_gold += b3_c_r_gold
                total_b3_p_gold += b3_p_gold
                total_b3_r_gold += b3_r_gold
                total_error_link_se += error_link_se
                total_error_link_p += error_link_p
                total_error_link_ee += error_link_ee

                total_rel_p_gold += link_rel_p_gold
                total_rel_r_gold += link_rel_r_gold
                total_rel_c_gold += link_rel_c_gold
                total_error_rel_ne += error_rel_ne
                total_error_rel_te += error_rel_te

                total_ground_c += batch_ground_c
                total_ground_p += batch_ground_p
                total_ground_r += batch_ground_r
                total_error_gro_be += error_gro_be
                total_error_gro_te += error_gro_te
        ent_precious, ent_recall, ent_f1 = self.calu_res(total_p, total_r, total_c)
        
        ground_precious, ground_recall, ground_f1 = self.calu_res(total_ground_p, total_ground_r, total_ground_c)

        rel_precious_gold, rel_recall_gold, rel_f1_gold = self.calu_res(total_rel_p_gold, total_rel_r_gold, total_rel_c_gold)

        ceaf_precious_gold, ceaf_recall_gold, ceaf_f1_gold = self.calu_res(total_ceaf_p_gold, total_ceaf_r_gold, total_ceaf_c_gold)
        muc_precious_gold, muc_recall_gold, muc_f1_gold = self.calu_res(total_muc_p_gold, total_muc_r_gold, total_muc_c_gold)
        b3_precious_gold, b3_recall_gold, b3_f1_gold = self.calu_res(total_b3_p_gold, total_b3_r_gold, total_b3_c_p_gold, total_b3_c_r_gold)

        Title = "DEV" if not is_test else "TEST"
        ent_e_1, ent_e_2 = error(total_error_ent_be, total_error_ent_te, total_p, "bourd", "type")
        link_e_1, link_e_2 = error(total_error_link_se, total_error_link_ee, total_error_link_p, "queshi", "cuowu")
        rel_e_1, rel_e_2 = error(total_error_rel_te, total_error_rel_ne, total_rel_p_gold, "type", "notype")
        gro_e_1, gro_e_2 = error(total_error_gro_be, total_error_gro_te, total_ground_p, "bourd", "type")

        table = pt.PrettyTable([f"Gold {Title} {epoch}", "Precision", "Recall", "F1"])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [ent_precious, ent_recall, ent_f1]])
        table.add_row(["EntityError"] + [ent_e_1, ent_e_2, 0])
        table.add_row(["MUC"] + ["{:3.4f}".format(x) for x in [muc_precious_gold, muc_recall_gold, muc_f1_gold]])
        table.add_row(["CEAF"] + ["{:3.4f}".format(x) for x in [ceaf_precious_gold, ceaf_recall_gold, ceaf_f1_gold]])
        table.add_row(["B3"] + ["{:3.4f}".format(x) for x in [b3_precious_gold, b3_recall_gold, b3_f1_gold]])
        table.add_row(["Link"] + ["{:3.4f}".format(x) for x in [np.mean([muc_precious_gold, ceaf_precious_gold, b3_precious_gold]), np.mean([muc_recall_gold, ceaf_recall_gold, b3_recall_gold]), np.mean([muc_f1_gold, ceaf_f1_gold, b3_f1_gold])]])
        table.add_row(["LinkError"] + [link_e_1, link_e_2, 0])
        table.add_row(["Relation"] + ["{:3.4f}".format(x) for x in [rel_precious_gold, rel_recall_gold, rel_f1_gold]])
        table.add_row(["RelationError"] + [rel_e_1, rel_e_2, 0])
        table.add_row(["Grounding"] + ["{:3.4f}".format(x) for x in [ground_precious, ground_recall, ground_f1]])
        table.add_row(["GroundingError"] + [gro_e_1, gro_e_2, 0])
        table.add_row(["Avg."] + ["{:3.4f}".format(x) for x in [np.mean([ent_precious, muc_precious_gold, ceaf_precious_gold, b3_precious_gold, rel_precious_gold, ground_precious]), np.mean([ent_recall, muc_recall_gold, ceaf_recall_gold, b3_recall_gold, rel_recall_gold, ground_recall]), np.mean([ent_f1, muc_f1_gold, ceaf_f1_gold, b3_f1_gold, rel_f1_gold, ground_f1])]])

        logger.info("\n{}".format(table))
        return np.mean([ent_f1, muc_f1_gold, ceaf_f1_gold, b3_f1_gold, rel_f1_gold, ground_f1])
    

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

def seed_torch(seed=3306):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.json')
    parser.add_argument('--gpu_id', type=int)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)
    parser.add_argument('--warm_factor', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--max_sequence_length', type=int)
    parser.add_argument('--lang', type=str, choices=["mix", "zh", "en"])
    parser.add_argument('--text_hidden', type=int)
    parser.add_argument('--feature_hidden', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--lstm_hid_size', type=int)
    parser.add_argument('--is_miss', type=str2bool)
    parser.add_argument('--cluster_threshold', type=float)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--logPath', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--is_h', type=str2bool)

    args = parser.parse_args()
    
    config = config.Config(args)
    
    torch.autograd.set_detect_anomaly(True)
    seed_torch(config.seed)
    logger = utils.get_logger(config)
    logger.info(config)
    config.logger = logger

    if torch.cuda.is_available():
        device = f"cuda:{config.gpu_id}"
    else:
        device = "cpu"
        
    config.device = device



    logger.info("Loading Data")
    datasets = data_loader.load_data_bert(config)

    train_loader, dev_loader, test_loader = (
        DataLoader(dataset=dataset,
                   batch_size=config.batch_size,
                   collate_fn=data_loader.collate_fn,
                   shuffle=i == 0,
                   num_workers=4)
        for i, dataset in enumerate(datasets)
    )

    updates_total = len(datasets[0]) // config.batch_size * config.epochs
    logger.info("Building Model")
    model = Model(config)

    model = model.to(device)
    # num_params = sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 3)

    trainer = Trainer(model)
    best_f1, best_f1_gold = 0, 0
    for i in range(config.epochs):
        logger.info("Epoch: {}".format(i))
        trainer.train(i, train_loader)
        all_avg_f1_gold = trainer.eval(i, dev_loader)
        trainer.eval(i, test_loader, is_test=True)
        if best_f1_gold <= all_avg_f1_gold:
            best_f1_gold = all_avg_f1_gold
            trainer.save(config.model_name)
    trainer.load(config.model_name)
    trainer.eval("Final", test_loader, True)
