from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import gzip
import random
from multiprocessing import Pool
import pickle
import math
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np
import os
from torch.utils.data.distributed import DistributedSampler
from copy import deepcopy

from transformers import T5Tokenizer, T5TokenizerFast
from tokenization import P5Tokenizer, P5TokenizerFast

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def ReadLineFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

    
class P5_MIND_Dataset(Dataset):
    def __init__(self, all_tasks, task_list, tokenizer, args, sample_numbers, mode ='train', split = 'MIND', rating_augment = False, sample_type = 'random'):


        self.all_tasks = all_tasks # sequential
        self.task_list = task_list
        self.tokenizer = tokenizer
        self.args = args
        self.sample_numbers = sample_numbers
        self.split = split
        self.rating_augment = rating_augment
        self.sample_type = sample_type
        self.mode = mode

        # upload corresponding data
        if self.mode == 'train':
            # {uid: [combined infor history]}
            self.his = pickle.load(
                open(os.path.join('./train_infor_his'), "rb"))
  
            # {(uid, pview, nview)]}
            self.interaction = pickle.load(
                open(os.path.join('./train_interaction'), "rb"))
   
           
        else:
            raise NotImplementedError


        self.total_length = 0
        self.datum_info = []
        self.compute_datum_info()
        
    def compute_datum_info(self):
        curr = 0
        for key in list(self.task_list.keys()):
            if key == 'sequential':
                self.total_length += len(self.interaction) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            else:
                raise NotImplementedError
            
    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        
        out_dict = {}
        out_dict['args'] = self.args
        loss_weight = 1.0
        
        datum_info_idx = self.datum_info[idx]
        assert datum_info_idx[0] == idx # index of the selected sample
        if len(datum_info_idx) == 3:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2] # row index of self.interaction
        else:
            raise NotImplementedError

        if task_name == 'sequential':

            user = self.interaction[datum_idx][0]
            pos_infor = self.interaction[datum_idx][1]
            neg_infor = self.interaction[datum_idx][2]
            
            #pos_infor = self.infor[pos_id]
            #neg_infor = self.infor[neg_id]
            
            # history
            history = self.his[user]
      

            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates) - 1)  # random choose the task index for task_candidates
            task_template = self.all_tasks['sequential'][task_candidates[task_idx]]
            assert task_template['task'] == 'sequential'

            if task_template['id'] == '1-1':

                p = random.random()
                
                #print('consider history length', self.args.history_length)
                his = history[-self.args.history_length: ]

                if p > 0.5:
                    # source_1 and source_2 are for content
                    source_1 = task_template['source1'].format(', '.join(his), pos_infor)
                    target_1 = task_template['target1'].format('yes')

                    source_2 = task_template['source1'].format(  ', '.join(his),  neg_infor)
                    target_2 = task_template['target1'].format('no')
                    
                  

                else:
                    source_1 = task_template['source1'].format(', '.join(his), neg_infor)
                    target_1 = task_template['target1'].format('no')

                    source_2 = task_template['source1'].format(', '.join(his), pos_infor)
                    target_2 = task_template['target1'].format('yes')
                    
                 

            else:
                raise NotImplementedError

            
        else:
            raise NotImplementedError

        # for the first sample
        input_ids_1 = self.tokenizer.encode(source_1, padding = True, truncation = True, max_length = self.args.max_text_length)
        tokenized_text_1 = self.tokenizer.tokenize(source_1)
        whole_word_ids_1 = self.calculate_whole_word_ids(tokenized_text_1, input_ids_1)
        assert len(whole_word_ids_1) == len(input_ids_1)
        target_ids_1 = self.tokenizer.encode(target_1, padding = True, truncation = True, max_length = self.args.gen_max_length)

        # for the second sample
        input_ids_2 = self.tokenizer.encode(source_2, padding = True, truncation = True, max_length = self.args.max_text_length)
        tokenized_text_2 = self.tokenizer.tokenize(source_2)
        whole_word_ids_2 = self.calculate_whole_word_ids(tokenized_text_2, input_ids_2)
        assert len(whole_word_ids_2) == len(input_ids_2)
        target_ids_2 = self.tokenizer.encode(target_2, padding = True, truncation = True, max_length = self.args.gen_max_length)

        
        # for the first sample
        out_dict['input_ids_1'] = torch.LongTensor(input_ids_1)
        out_dict['input_length_1'] = len(input_ids_1)
        out_dict['source_text_1'] = source_1
        out_dict['tokenized_text_1'] = tokenized_text_1
        out_dict['whole_word_ids_1'] = torch.LongTensor(whole_word_ids_1)

        out_dict['target_ids_1'] = torch.LongTensor(target_ids_1)
        out_dict['target_length_1'] = len(target_ids_1)
        out_dict['target_text_1'] = target_1
        out_dict['loss_weight_1'] = loss_weight

        # for the second sample
        out_dict['input_ids_2'] = torch.LongTensor(input_ids_2)
        out_dict['input_length_2'] = len(input_ids_2)
        out_dict['source_text_2'] = source_2
        out_dict['tokenized_text_2'] = tokenized_text_2
        out_dict['whole_word_ids_2'] = torch.LongTensor(whole_word_ids_2)

        out_dict['target_ids_2'] = torch.LongTensor(target_ids_2)
        out_dict['target_length_2'] = len(target_ids_2)
        out_dict['target_text_2'] = target_2
        out_dict['loss_weight_2'] = loss_weight
        
    

        out_dict['task'] = 'sequential'

        return out_dict
    
    def calculate_whole_word_ids(self, tokenized_text, input_ids):
        whole_word_ids = []
        curr = 0
        for i in range(len(tokenized_text)):
            if tokenized_text[i].startswith('▁'):
                curr += 1
                whole_word_ids.append(curr)
            else:
                whole_word_ids.append(curr)
        last_item = whole_word_ids[len(input_ids) - 2]
        return whole_word_ids[:len(input_ids) - 1] + [0] ## the added [0] is for </s>


    def collate_fn(self, batch):

        args = self.args
        batch_entry = {}

        # len(batch) = 16 -> 32 samples
        # batch = [idx1, idx2,...] -> for each idx, generate two separate sentences
        # 合并
        B = len(batch)  * 2

        max_input_length_1 = max(entry['input_length_1'] for entry in batch)
        max_input_length_2 = max(entry['input_length_2'] for entry in batch)
      
        S_W_L = max(max_input_length_1, max_input_length_2)

        max_target_length_1 = max(entry['target_length_1'] for entry in batch)
        max_target_length_2 = max(entry['target_length_2'] for entry in batch)
      
        T_W_L = max(max_target_length_1, max_target_length_2)

        input_ids = torch.ones(B, S_W_L, dtype = torch.long) * self.tokenizer.pad_token_id
        whole_word_ids = torch.ones(B, S_W_L, dtype = torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype = torch.long) * self.tokenizer.pad_token_id

        loss_weights = torch.ones(B, dtype = torch.float)

        # i.e. batch_size = 16 -> want the output size = 32 samples
        tasks = []
        source_text_1 = []
        source_text_2 = []
      
        tokenized_text_1 = []
        tokenized_text_2 = []
        
        target_text_1 = []
        target_text_2 = []
       
        for i, entry in enumerate(batch):
            # 先放入第一个sample/sentence
            input_ids[i, :entry['input_length_1']] = entry['input_ids_1']
            whole_word_ids[i, :entry['input_length_1']] = entry['whole_word_ids_1']
            target_ids[i, :entry['target_length_1']] = entry['target_ids_1']

            # 放入第二个sample
            input_ids[i + len(batch), :entry['input_length_2']] = entry['input_ids_2']
            whole_word_ids[i + len(batch), :entry['input_length_2']] = entry['whole_word_ids_2']
            target_ids[i + len(batch), :entry['target_length_2']] = entry['target_ids_2']
            
         
            if 'task' in entry:
                tasks.append(entry['task']) # 16个

            if 'source_text_1' in entry:
                source_text_1.append(entry['source_text_1']) # length = len(batch)
            if 'source_text_2' in entry:
                source_text_2.append(entry['source_text_2']) # length = len(batch)
       

            if 'tokenized_text_1' in entry:
                tokenized_text_1.append(entry['tokenized_text_1'])
            if 'tokenized_text_2' in entry:
                tokenized_text_2.append(entry['tokenized_text_2'])
           
            if 'target_text_1' in entry:
                target_text_1.append(entry['target_text_1'])
            if 'target_text_2' in entry:
                target_text_2.append(entry['target_text_2'])
         
            if 'loss_weight_1' in entry:
                if entry['target_length_1'] > 0:
                    loss_weights[i] = entry['loss_weight_1'] / entry['target_length_1']
                else:
                    loss_weights[i] = entry['loss_weight_1']

           
            if 'loss_weight_2' in entry:
                if entry['target_length_2'] > 0:
                    loss_weights[i + len(batch)] = entry['loss_weight_2']/entry['target_length_2']
                else:
                    loss_weights[i + len(batch)] = entry['loss_weight_2']

           

        # tasks -> length = 32, 0 和16 corresponds to the same user
        task = tasks  + tasks 
        source_text = source_text_1 + source_text_2 
        target_text = target_text_1 + target_text_2 

        # create a batch
        assert 't5' in args.backbone
        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100

        # batch_entry = 2*len(batch)
        batch_entry['task'] = task
        batch_entry['source_text'] = source_text
        batch_entry['target_text'] = target_text

        batch_entry['input_ids'] = input_ids
        batch_entry['whole_word_ids'] = whole_word_ids
        batch_entry['target_ids'] = target_ids

        batch_entry['loss_weights'] = loss_weights
        
        #print(batch_entry['source_text'])
        return batch_entry


# validation/test dataloader
class P5_val_Dataset(Dataset):
    def __init__(self, all_tasks, task_list, tokenizer, args, sample_numbers, mode = 'val', split = 'MIND',
                 rating_augment = False, sample_type = 'random'):
        self.all_tasks = all_tasks  # sequential
        self.task_list = task_list
        self.tokenizer = tokenizer
        self.args = args
        self.sample_numbers = sample_numbers
        self.split = split
        self.rating_augment = rating_augment
        self.sample_type = sample_type
        self.mode = mode


        # upload corresponding data
        if self.mode == 'val':
            # history {raw user id: ['list of title history']}
            self.his = pickle.load(
                open(os.path.join('./his'), "rb"))
  
            # {(uid, pview, nview)]}
            self.interaction = pickle.load(
                open(os.path.join('./interaction'), "rb"))
             
            self.infor = pickle.load(
                open(os.path.join('./news_infor'), "rb"))
             
        
        else:
            raise NotImplementedError


        self.total_length = 0
        self.datum_info = []
        self.compute_datum_info()

    def compute_datum_info(self):
        curr = 0
        for key in list(self.task_list.keys()):
            if key == 'sequential':
                self.total_length += len(self.interaction) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            else:
                raise NotImplementedError

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args
        loss_weight = 1.0

        datum_info_idx = self.datum_info[idx]
        assert datum_info_idx[0] == idx  # index of the selected sample
        if len(datum_info_idx) == 3:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]  # row index of self.interaction
        else:
            raise NotImplementedError

        if task_name == 'sequential':

            raw_user = self.interaction[datum_idx][0]
            impress_id = self.interaction[datum_idx][1]
            item_id = self.interaction[datum_idx][2]
            item_infor = self.infor[item_id]
            response = self.interaction[datum_idx][3]
            

            # history
            history = self.his[raw_user]
            
            
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates) - 1)  # random choose the task index for task_candidates
            task_template = self.all_tasks['sequential'][task_candidates[task_idx]]
            

            assert task_template['task'] == 'sequential'
            if task_template['id'] == '1-1':
                
                his = history[-self.args.history_length:]
            
                source_1 = task_template['source1'].format(', '.join(his), item_infor)
                target_1 = task_template['target1'].format(response)
                

            else:
                raise NotImplementedError
            

        else:
            raise NotImplementedError

        # for the first sample
        input_ids_1 = self.tokenizer.encode(source_1, padding=True, truncation=True, max_length=self.args.max_text_length)
        tokenized_text_1 = self.tokenizer.tokenize(source_1)
        whole_word_ids_1 = self.calculate_whole_word_ids(tokenized_text_1, input_ids_1)
        assert len(whole_word_ids_1) == len(input_ids_1)
        target_ids_1 = self.tokenizer.encode(target_1, padding=True, truncation=True,max_length=self.args.gen_max_length)
      
        # for the first sample
        out_dict['input_ids_1'] = torch.LongTensor(input_ids_1)
        out_dict['input_length_1'] = len(input_ids_1)
        out_dict['source_text_1'] = source_1
        out_dict['tokenized_text_1'] = tokenized_text_1
        out_dict['whole_word_ids_1'] = torch.LongTensor(whole_word_ids_1)

        out_dict['target_ids_1'] = torch.LongTensor(target_ids_1)
        out_dict['target_length_1'] = len(target_ids_1)
        out_dict['target_text_1'] = target_1
        out_dict['loss_weight_1'] = loss_weight
        
        out_dict['item_id'] = item_id
        out_dict['impress_id'] = impress_id
        out_dict['user_id'] = raw_user
        
        out_dict['task'] = 'sequential'

        return out_dict

    def calculate_whole_word_ids(self, tokenized_text, input_ids):
        whole_word_ids = []
        curr = 0
        for i in range(len(tokenized_text)):
            if tokenized_text[i].startswith('▁'):
                curr += 1
                whole_word_ids.append(curr)
            else:
                whole_word_ids.append(curr)
        last_item = whole_word_ids[len(input_ids) - 2]
        return whole_word_ids[:len(input_ids) - 1] + [0]  ## the added [0] is for </s>

    def collate_fn(self, batch):
        args = self.args

        batch_entry = {}
        B = len(batch)

        S_W_L = max(entry['input_length_1'] for entry in batch)
        T_W_L = max(entry['target_length_1'] for entry in batch)
       
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        whole_word_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        loss_weights = torch.ones(B, dtype=torch.float)

        # i.e. batch_size = 16 -> want the output size = 32 samples
        tasks = []
        user_id = []
        impress_id = []
        item_id = []
        
        source_text_1 = []
        tokenized_text_1 = []
        target_text_1 = []
        

        for i, entry in enumerate(batch):
            # 先放入第一个sample/sentence
            input_ids[i, :entry['input_length_1']] = entry['input_ids_1']
            whole_word_ids[i, :entry['input_length_1']] = entry['whole_word_ids_1']
            target_ids[i, :entry['target_length_1']] = entry['target_ids_1']


            if 'task' in entry:
                tasks.append(entry['task']) # 16个

            if 'source_text_1' in entry:
                source_text_1.append(entry['source_text_1']) # length = len(batch)
    

            if 'tokenized_text_1' in entry:
                tokenized_text_1.append(entry['tokenized_text_1'])
        

            if 'target_text_1' in entry:
                target_text_1.append(entry['target_text_1'])
          

            # for a certain index in batch
            if 'loss_weight_1' in entry:
                if entry['target_length_1'] > 0:
                    loss_weights[i] = entry['loss_weight_1'] / entry['target_length_1']
                else:
                    loss_weights[i] = entry['loss_weight_1']

    

            if 'impress_id' in entry:
                impress_id.append(entry['impress_id'])

            if 'user_id' in entry:
                user_id.append(entry['user_id'])

            if 'item_id' in entry:
                item_id.append(entry['item_id'])

  
        task = tasks
        source_text = source_text_1 
        target_text = target_text_1 

        # create a batch
        assert 't5' in args.backbone
        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100

        batch_entry['task'] = task
        batch_entry['source_text'] = source_text
        batch_entry['target_text'] = target_text
        batch_entry['user_id'] = user_id 
        batch_entry['item_id'] = item_id
        batch_entry['impress_id'] = impress_id 

        batch_entry['input_ids'] = input_ids
        batch_entry['whole_word_ids'] = whole_word_ids
        batch_entry['target_ids'] = target_ids

        batch_entry['loss_weights'] = loss_weights

        #print(batch_entry['source_text'])
        return batch_entry

    
    


def get_loader(args, task_list, sample_numbers, split = 'MIND', mode = 'train',
               batch_size = 16, workers = 4, distributed = False):

    if 't5' in args.backbone:
        # add user_id_ and item_id to the additional special token besides extra_id_ tokens
        tokenizer = P5Tokenizer.from_pretrained(
            args.backbone, 
            max_length = args.max_text_length,
            do_lower_case = args.do_lower_case)


    from MIND_templates import all_tasks as task_templates

    if mode == 'train':
        #args.LOSSES_NAME.append('pair_loss')
        #args.LOSSES_NAME.append('total_loss')
        dataset = P5_MIND_Dataset(
            task_templates,
            task_list,
            tokenizer, # t5-small
            args,
            sample_numbers,
            mode = mode,
            split = split,
            rating_augment = False
        )

    if mode == 'val':
        dataset = P5_val_Dataset(
            task_templates,
            task_list,
            tokenizer,
            args,
            sample_numbers,
            mode = mode,
            split = split,
            rating_augment = False
        )
    
    ###### test/inference
    if mode == 'test':
        dataset = P5_test_Dataset(
            task_templates,
            task_list,
            tokenizer,
            args,
            sample_numbers,
            mode = mode,
            split = split,
            rating_augment = False
        )
    
    
    

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers = workers, pin_memory = True, sampler = sampler,
            collate_fn = dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size = batch_size,
            num_workers = workers, pin_memory=True,
            sampler = sampler,
            shuffle = None if (sampler is not None) else False,
            collate_fn = dataset.collate_fn,
            drop_last = False)
        
    return loader
