import logging
import torch
from torch.nn import functional as F
import sentencepiece as spm
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel, XLMRobertaForMaskedLM
from transformers import AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import pickle
from models import HardMasked



if __name__ == "__main__":
    detector_path = './model/Detector967.pkl'
    detector_tokenizer_path = './model/spm_tokenizer.model'

    MaskedLM = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-base')

    maskedlm_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

    detector_tokenizer = spm.SentencePieceProcessor(detector_tokenizer_path, )

    detector = torch.load(detector_path)

    model = HardMasked(detector, MaskedLM, detector_tokenizer, maskedlm_tokenizer, 'cpu')
    
    s = 'Tôi vẫn luôn iu cô ấy với hết tấm lòng của mk'
    print(model(s))