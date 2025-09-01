# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


class IMDBDatasetHelper:
    def __init__(self, tokenizer, split, batch_size, max_seq_len, num_workers):
        self.tokenizer = tokenizer
        self.split = split
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.pad_token_id = tokenizer.pad_token_id
        assert split in [
            "train",
            "test",
        ], "IMDB Dataset has only 'train' and 'test' split."

        imdb_dataset = load_dataset("stanfordnlp/imdb")[split]
        self.data_loader = DataLoader(
            imdb_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_batch,
            num_workers=num_workers,
        )

    def get_dataloader(self):
        return self.data_loader

    def collate_batch(self, batch_data):
        batched_input_ids = []
        batched_attn_mask = []
        batched_token_type_ids = []
        batched_labels = []
        for data in batch_data:
            tokenized_data = self.tokenizer(
                data["text"], truncation=True, max_length=self.max_seq_len
            )
            batched_input_ids.append(
                torch.tensor(tokenized_data["input_ids"], dtype=torch.int32)
            )
            batched_attn_mask.append(
                torch.tensor(tokenized_data["attention_mask"], dtype=torch.int32)
            )
            batched_token_type_ids.append(
                torch.tensor(tokenized_data["token_type_ids"], dtype=torch.int32)
            )
            batched_labels.append(torch.tensor([data["label"]], dtype=torch.int32))
        batched_input_ids = pad_sequence(
            batched_input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        batched_attn_mask = pad_sequence(
            batched_attn_mask, batch_first=True, padding_value=0
        )
        batched_token_type_ids = pad_sequence(
            batched_token_type_ids, batch_first=True, padding_value=0
        )
        batched_labels = torch.cat(batched_labels)
        return {
            "input_ids": batched_input_ids,
            "attention_mask": batched_attn_mask,
            "token_type_ids": batched_token_type_ids,
            "label": batched_labels,
        }
