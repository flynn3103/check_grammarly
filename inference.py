import os
import json
import logging
from typing import Mapping

import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn 

import sentencepiece as spm
from seqeval.metrics import precision_score as seq_precision, recall_score as seq_recall, f1_score as seq_f1
from transformers import AutoTokenizer, XLMRobertaModel, XLMRobertaForMaskedLM
from transformers import AdamW, get_linear_schedule_with_warmup


from tqdm.notebook import tqdm
from easydict import EasyDict
import gc
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pickle

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    detector_path = '/content/Detector967.pkl'
    detector_tokenizer_path = '/content/spm_tokenizer.model'

    MaskedLM = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-base')

    maskedlm_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

    detector_tokenizer = spm.SentencePieceProcessor(detector_tokenizer_path, )

    detector = torch.load(detector_path)

    model = HardMasked(detector, MaskedLM, detector_tokenizer, maskedlm_tokenizer, 'cuda')
    
    s = 'Tôi vẫn luôn iu cô ấy với hết tấm lòng của mk'
    model(s)