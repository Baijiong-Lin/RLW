import random, logging, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from processors.utils_tag import convert_examples_to_features_tag, get_labels, read_examples_from_file

from processors.utils_sc import convert_examples_to_features_sc
from processors.xnli import XnliProcessor
from processors.pawsx import PawsxProcessor

from transformers import BertTokenizer

from transformers.data.processors.squad import SquadV1Processor, squad_convert_examples_to_features
# logger = logging.getLogger(__name__)

# Token Classification: NER(panx), POS(udpos)
def DataloaderTC(lang_list,
                model_name_or_path,
                model_type,
                mode_list,
                data_dir,
                max_seq_length,
                batch_size, small_train=None):
    lang2id = None # if model_type != 'xlm'
    labels = get_labels(os.path.join(data_dir, 'labels.txt'))
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=False)
    pad_token_label_id = nn.CrossEntropyLoss().ignore_index
    
    dataloader = {}
    iter_dataloader = {}
    
    for lang in lang_list:
        dataloader[lang] = {}
        iter_dataloader[lang] = {}
        for mode in mode_list:
            if mode == 'train' and small_train is not None:
                cached_features_file = os.path.join(data_dir, lang, "cached_feature_{}_{}_{}_{}".format(mode,
                                            list(filter(None, model_name_or_path.split("/"))).pop(),
                                            str(max_seq_length), small_train))
            else:
                cached_features_file = os.path.join(data_dir, lang, "cached_feature_{}_{}_{}".format(mode,
                                        list(filter(None, model_name_or_path.split("/"))).pop(),
                                        str(max_seq_length)))
            try:
                features_lg = torch.load(cached_features_file)
                print("Loading features from cached file {}".format(cached_features_file))
            except:
                data_file = os.path.join(data_dir, lang, "{}.{}".format(mode, model_name_or_path))
                print("Creating features from dataset file at {} in language {}".format(cached_features_file, lang))
                examples = read_examples_from_file(data_file, lang, lang2id)
                features_lg = convert_examples_to_features_tag(examples, labels, max_seq_length, tokenizer,
                                        cls_token_at_end=bool(model_type in ["xlnet"]),
                                        cls_token=tokenizer.cls_token,
                                        cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                                        sep_token=tokenizer.sep_token,
                                        sep_token_extra=bool(model_type in ["roberta", "xlmr"]),
                                        pad_on_left=bool(model_type in ["xlnet"]),
                                        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                        pad_token_segment_id=4 if model_type in ["xlnet"] else 0,
                                        pad_token_label_id=pad_token_label_id,
                                        lang=lang
                                        )
                if mode == 'train' and small_train is not None:
                    random.shuffle(features_lg)
                    features_lg = features_lg[:small_train]
                torch.save(features_lg, cached_features_file)
                
            
            all_input_ids = torch.tensor([f.input_ids for f in features_lg], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in features_lg], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in features_lg], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_ids for f in features_lg], dtype=torch.long)

            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            sampler = RandomSampler(dataset) if mode in ['train'] else SequentialSampler(dataset)
            drop_last = True if mode in ['train'] else False
            dataloader[lang][mode] = DataLoader(dataset, sampler=sampler, 
                                                batch_size=batch_size, 
                                                num_workers=2, 
                                                pin_memory=True,
                                                drop_last=drop_last)
            iter_dataloader[lang][mode] = iter(dataloader[lang][mode])
    return dataloader, iter_dataloader, labels


# Sequence Classification: XNIL, PAWS-X
def DataloaderSC(lang_list,
                model_name_or_path,
                model_type,
                mode_list,
                data_dir,
                max_seq_length,
                batch_size, small_train=None):
    lang2id = None # if model_type != 'xlm'
    if 'pawsx' in data_dir.split('/')[-1]:
        processor = PawsxProcessor()
    elif 'xnli' in data_dir.split('/')[-1]:
        processor = XnliProcessor()
    else:
        raise('no support')
    output_mode = "classification"
    label_list = processor.get_labels()
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=False)
    
    dataloader = {}
    iter_dataloader = {}
    
    for lang in lang_list:
        dataloader[lang] = {}
        iter_dataloader[lang] = {}
        for mode in mode_list:
            if mode == 'train' and small_train is not None:
                cached_features_file = os.path.join(data_dir, "cached_feature_{}_{}_{}_{}_{}".format(mode, lang,
                                            list(filter(None, model_name_or_path.split("/"))).pop(),
                                            str(max_seq_length), small_train))
            else:
                cached_features_file = os.path.join(data_dir, "cached_feature_{}_{}_{}_{}".format(mode, lang,
                                            list(filter(None, model_name_or_path.split("/"))).pop(),
                                            str(max_seq_length)))
            try:
                features_lg = torch.load(cached_features_file)
                print("Loading features from cached file {}".format(cached_features_file))
            except:
                print("Creating features from dataset file at {} in language {} and mode {}".format(cached_features_file, lang, mode))
                examples = processor.get_examples(data_dir, language=lang, split=mode)
                features_lg = convert_examples_to_features_sc(
                                            examples,
                                            tokenizer,
                                            label_list=label_list,
                                            max_length=max_seq_length,
                                            output_mode=output_mode,
                                            pad_on_left=False,
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=0,
                                            lang2id=lang2id,
                                        )
                if mode == 'train' and small_train is not None:
                    random.shuffle(features_lg)
                    features_lg = features_lg[:small_train]
                torch.save(features_lg, cached_features_file)
                
       
            all_input_ids = torch.tensor([f.input_ids for f in features_lg], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in features_lg], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features_lg], dtype=torch.long)
            all_labels = torch.tensor([f.label for f in features_lg], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

            sampler = RandomSampler(dataset) if mode in ['train'] else SequentialSampler(dataset)
            drop_last = True if mode in ['train'] else False
            dataloader[lang][mode] = DataLoader(dataset, 
                                                sampler=sampler, 
                                                batch_size=batch_size, 
                                                num_workers=2, 
                                                pin_memory=True,
                                                drop_last=drop_last)
            iter_dataloader[lang][mode] = iter(dataloader[lang][mode])
    return dataloader, iter_dataloader, label_list