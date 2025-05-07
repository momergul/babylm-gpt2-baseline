# File: data_utils.py
# -------------------
# Function for dataset loading, construction and saving + collation functions

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

import math
import random
import os
from tqdm import tqdm
import pickle

MODEL_EOS = 50256
TRAIN_PATH_100M = 'data/text_data/clean_train_100M'
TRAIN_PATH_10M = 'data/text_data/clean_train_10M'
DATASETS = ['bnc_spoken', 'childes', 'gutenberg', 'open_subtitles', 'simple_wiki', 'switchboard']

class FullBabyLMDataset(Dataset):

    def __init__(self, cfg):
        # First load the tokenizer
        self.processor = AutoTokenizer.from_pretrained("gpt2")

        # Tokenize, split and reconstruct each dataset
        self.data = []
        dataset_folder = TRAIN_PATH_100M if cfg["training_type"] == "strict" else TRAIN_PATH_10M

        for dataset in DATASETS:
            # Load all text in dset
            dataset_path = os.path.join(dataset_folder, f'{dataset}.train')
            with open(dataset_path, 'r') as f:
                all_text = ' '.join(f.readlines())
            print(f'Opened {dataset_path}')

            # Process full text into tokens
            tokenized_dataset = self.processor(text=[all_text])['input_ids'][0]
            print(f'Tokenized {dataset_path}; {len(tokenized_dataset)} tokens total')

            # Chunk and add
            chunk_size = cfg["datapoint_length"]
            num_chunks = math.ceil(len(tokenized_dataset) / chunk_size)
            for curr_chunk in tqdm(range(num_chunks)):
                start = curr_chunk * chunk_size
                end = (curr_chunk+1) * chunk_size
                chunk_tokens = tokenized_dataset[start:end]
                self.data.append(chunk_tokens)
            print(f"Chunked {dataset_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.LongTensor([MODEL_EOS] + self.data[idx] + [MODEL_EOS])

## General utilities ##
def load_babylm_data(cfg):
    # Get the overall BabyLM dataset to extract data from (behavior may vary)
    num_words = "100M" if cfg["training_type"] == "strict" else "10M"
    filename = os.path.join('data/text_data/cached_train', f'train_gpt2_{num_words}.pkl')
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            full_babylm_dset = pickle.load(f)
    else:
        full_babylm_dset = FullBabyLMDataset(cfg)
        with open(filename, 'wb') as f:
            pickle.dump(full_babylm_dset, f)

    dataloader = DataLoader(full_babylm_dset, batch_size=cfg["batch_size"],
                            shuffle=True, collate_fn=collate_fn)
    return dataloader

def collate_fn(batch):
    tokens = pad_sequence([item for item in batch], padding_value=MODEL_EOS, batch_first=True)
    input_tokens = tokens[:, :-1]
    target_tokens = tokens[:, 1:]
    target_mask = input_tokens != MODEL_EOS
    target_mask[:, 0] = 1

    return input_tokens, target_tokens, target_mask
    
