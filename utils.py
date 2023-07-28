import copy
import torch
import transformers
from typing import Sequence, Dict
from datasets import Dataset


IGNORE_INDEX = -100
PAD_TOKEN = "[PAD]"
EOS_TOKEN = "</s>"
BOS_TOKEN = "</s>"
UNK_TOKEN = "</s>"

def tokenize_QA_for_llm(
        ds: Dataset, tokenizer,
        workers: int = 2, batch_size=1024,
) -> Dataset:
    def tokenize(example):
        question = example['question'].strip()
        answer = example['answer'].strip()
        tokenized_inputs = tokenizer(
            question + answer,
            return_tensors="pt",
            padding="longest",
            max_length=2048,
            truncation=True,
        )
        tokenized_questions = tokenizer(
            question,
            return_tensors="pt",
            padding="longest",
            max_length=2048,
            truncation=True,
        )
        input_ids = tokenized_inputs.input_ids[0]
        label_ids = copy.deepcopy(input_ids)
        q_len = tokenized_questions.input_ids[0].ne(tokenizer.pad_token_id).sum().item()
        label_ids[:q_len] = IGNORE_INDEX
        return dict(
            input_ids=input_ids,
            labels=label_ids,
        )
    
    ds_tokenized = ds.map(
        tokenize,
        remove_columns=ds.column_names,
        num_proc=workers,
    )
    ds_tokenized.set_format(type='torch')
    return ds_tokenized


def resize_token_embedding(
    special_tokens_dict, tokenizer, model,
):
    """
    Simply initialize new token embeddings with the mean value of existing tokens so that
    the difference of KL-divergence is small.
    see https://nlp.stanford.edu/~johnhew/vocab-expansion.html
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_mean = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_mean = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_mean
        output_embeddings[-num_new_tokens:] = output_embeddings_mean


class DataCollatorForLLM(object):

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
