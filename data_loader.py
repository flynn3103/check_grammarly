import sentencepiece as spm
from seqeval.metrics import precision_score as seq_precision, recall_score as seq_recall, f1_score as seq_f1
from transformers import AutoTokenizer, XLMRobertaModel, XLMRobertaForMaskedLM
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import os
import torch
import numpy as np
import torch.nn as nn 
from torch.nn import functional as F
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.notebook import tqdm
from easydict import EasyDict
import gc
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.optim import Adam
import pickle

logger = logging.getLogger(__name__)

class InputExample(object):
    r"""
    One single example
    Args:
    example_id: str,  unique id for this example
    raw_text:list of words,  raw text contain misspelling words
    onehot_label:np array,  onehot array indicate position of the wrong words
    text_label:list of words,  ground true text label of raw_text
    """
    def __init__(self,example_id,  raw_text, onehot_labels= None, text_label= None):
        self.example_id = example_id
        self.raw_text = raw_text
        self.onehot_labels = onehot_labels
        self.text_label = text_label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    r"""
    Features of one single example
    Args:
    input_ids: ids of input raw text, after roberta tokenizer
    attention_mask: attention mask of input
    onehot_label: onehot array indicate position of wrong value

    """
    def __init__(self,input_ids, attention_mask, onehot_labels, output_ids ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.onehot_labels = onehot_labels
        self.output_ids = output_ids
        
    def __repr__(self):
        return str(self.to_json_string())
    
    def to_dict(self):
        """Serializes this instanc eto a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
    
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent = 2, sort_keys = True) + '\n'

class JointProcessor(object):
    
    def __init__(self, args, ):
        self.args = args
        self.raw_text_file = 'raw_text.txt'
        self.onehot_labels_file = 'onehot_label.txt'
        self.text_label_file = 'text_label.txt'

    @classmethod
    # read the whole file into a list, each elements is a sentence
    def _read_file(cls, input_file, ):
        with open(input_file, 'r', encoding = 'utf-8') as f:
            lines = []
            for line in f.readlines():
                lines.append(line.rstrip())

            return lines

    def _create_examples(self, texts, onehot_labels, text_labels, set_type):
        """create examples for training and dev set"""

        examples = []
        for i , (text, onehot, text_label) in enumerate(zip(texts, onehot_labels, text_labels)):
            example_id = "%s-%s"%(set_type, i)
            # 1. input_raw_text
            words = text.split()
            # 2. onehot_label
            onehot = np.array([int(x) for x in onehot.split()], dtype = int)
            # 3. text label
            text_label = text_label.split()
            #assert len(words) == len(onehot) == len(text_label)
            examples.append(InputExample(example_id, words, onehot, text_label))
        return examples

    def get_examples(self,mode):
        """
        Args:
        mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir,  )
        logger.info('Looking at {}'.format(data_path))
        return self._create_examples(texts = self._read_file(os.path.join(data_path, self.raw_text_file)),
                                    onehot_labels = self._read_file(os.path.join(data_path, self.onehot_labels_file)),
                                    text_labels = self._read_file(os.path.join(data_path, self.text_label_file)), set_type = mode)
        
        

def convert_examples_to_features(examples, max_seq_len, tokenizer , pad_token_label_id= 0):

    ignore_token = ' ?? '
    pad_token_id = 0
    features = []
    for ex_index, example in tqdm(enumerate(examples)):
        if ex_index % 10000 == 0:
            logger.info('Writing example %d of %d'%(ex_index + 1, len(examples)))
        # tokenize word by word
        raw_word_tokens = []
        onehot_labels = []
        text_label_tokens = []

        for raw_word, onehot, label_word in zip(example.raw_text, example.onehot_labels, example.text_label):

            raw_word_token = tokenizer.encode(raw_word, out_type = int)

            text_label_token = tokenizer.encode(label_word, out_type = int)
            # all the subtoken of word will have the same onehot label as word
            

            if len(raw_word_token) > len(text_label_token):
                text_label_token += [pad_token_id]*(len(raw_word_token) - len(text_label_token))

            if len(raw_word_token) < len(text_label_token):
                raw_word_token += [pad_token_id]*(len(text_label_token) - len(raw_word_token))

            onehot_labels.extend([int(onehot)]*len(raw_word_token))     


            raw_word_tokens.extend(raw_word_token)

            text_label_tokens.extend(text_label_token)
            

        assert len(onehot_labels) == len(raw_word_tokens) == len(text_label_tokens), 'word tokens len does not match one hot labels'


        input_ids = raw_word_tokens[:max_seq_len]
        onehot_labels = onehot_labels[:max_seq_len]
        output_ids = text_label_tokens[:max_seq_len]

        padding_len = (max_seq_len - len(input_ids))
        input_ids = input_ids + [pad_token_id]*padding_len
        onehot_labels = onehot_labels + [pad_token_label_id]*padding_len
        output_ids = output_ids + [pad_token_id]*padding_len
 

        assert len(input_ids) == max_seq_len, "input_ids Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(onehot_labels) == max_seq_len, "onehot_labels Error with input length {} vs {}".format(len(onehot_labels), max_seq_len)
        


        if ex_index < 5:
            logger.info('Example %s'%example.example_id)
            logger.info('Input ids %s'%' '.join([str(x) for x in input_ids]))
            logger.info('One hot labels %s'%' '.join([str(x) for x in onehot_labels]))
            logger.info('Output ids %s'%' '.join([str(x) for x in output_ids]))


        features.append(InputFeatures(input_ids = input_ids,attention_mask = None, onehot_labels = onehot_labels, output_ids = output_ids ))


    return features

def load_and_cache_examples(args,tokenizer,  mode):
    processor = JointProcessor(args)
    # Loooking for cached file
    cached_features_file = os.path.join(args.data_dir,
                                        'cached_%s_%s_%s'%(mode, str(args.max_seq_len), 'hard_masked_data') )

    if os.path.exists(cached_features_file):
        logger.info('Loading cached features file from %s'%cached_features_file)
        features = torch.load(cached_features_file)

    else:
        # load raw data to InputFeatures
        logger.info('Loading data from %s'%args.data_dir)
        if mode == 'train':
            examples= processor.get_examples(mode = 'train')
        elif mode == 'dev':
            examples = processor.get_examples(mode = 'dev')
        elif mode == 'test':
            examples = processor.get_examples(mode = 'test')
        else:
            raise Exception('Only train, dev, test are accepted')

        pad_token_label_id = args.ignore_index
        features = convert_examples_to_features(examples, args.max_seq_len,tokenizer, pad_token_label_id)
        logger.info('Save features file to %s'%cached_features_file)
        torch.save(features, cached_features_file)
    
    #Convert features to tensordataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype = torch.long)
    all_onehot_labels = torch.tensor([f.onehot_labels for f in features], dtype = torch.long)
    all_output_ids = torch.tensor([f.output_ids for f in features], dtype= torch.long)

    dataset = torch.utils.data.TensorDataset(all_input_ids, all_onehot_labels, all_output_ids, )
    return dataset