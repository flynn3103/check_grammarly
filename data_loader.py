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
        
        


