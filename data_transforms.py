import sentencepiece as spm
import os
import numpy as np
import re
import time
from tqdm.notebook import tqdm
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import unidecode
import string
from tqdm.notebook import tqdm
from nltk.tokenize.treebank import TreebankWordDetokenizer
import logging
from utils import init_logger

nltk.download('punkt')
sentence_tokenizer  =  nltk.data.load('tokenizers/punkt/english.pickle')

init_logger()
logger = logging.getLogger(__name__)

class SynthesizeData(object):
    """
    Uitils class to create artificial miss-spelled words
    Args: 
        vocab_path: path to vocab file. Vocab file is expected to be a set of words, separate by ' ', no newline charactor.
    """
    def __init__(self, vocab_path,  ):

        self.vocab = open(vocab_path, 'r', encoding = 'utf-8').read().split()
        self.tokenizer = word_tokenize
        self.word_couples = [ ['sương', 'xương'], ['sĩ', 'sỹ'], ['sẽ', 'sẻ'], ['sã', 'sả'], ['sả', 'xả'], ['sẽ', 'sẻ'], ['mùi', 'muồi'], 
                        ['chỉnh', 'chỉn'], ['sữa', 'sửa'], ['chuẩn', 'chẩn'], ['lẻ', 'lẽ'], ['chẳng', 'chẵng'], ['cổ', 'cỗ'], 
                        ['sát', 'xát'], ['cập', 'cặp'], ['truyện', 'chuyện'], ['xá', 'sá'], ['giả', 'dả'], ['đỡ', 'đở'], 
                        ['giữ', 'dữ'], ['giã', 'dã'], ['xảo', 'sảo'], ['kiểm', 'kiễm'], ['cuộc', 'cục'], ['dạng', 'dạn'], 
                        ['tản', 'tảng'], ['ngành', 'nghành'], ['nghề', 'ngề'], ['nổ', 'nỗ'], ['rảnh', 'rãnh'], ['sẵn', 'sẳn'], 
                        ['sáng', 'xán'], ['xuất', 'suất'], ['suôn', 'suông'], ['sử', 'xử'], ['sắc', 'xắc'], ['chữa', 'chửa'], 
                        ['thắn', 'thắng'], ['dỡ', 'dở'], ['trải', 'trãi'], ['trao', 'trau'], ['trung', 'chung'], ['thăm', 'tham'], 
                        ['sét', 'xét'], ['dục', 'giục'], ['tả', 'tã'],['sông', 'xông'], ['sáo', 'xáo'], ['sang', 'xang'], 
                        ['ngã', 'ngả'], ['xuống', 'suống'], ['xuồng', 'suồng'] ]


        self.vn_alphabet = ['a',  'ă', 'â', 'b' ,'c', 'd', 'đ', 'e','ê','g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'ô', 'ơ', 'p', 'q', 'r', 's', 't', 'u', 'ư', 'v', 'x', 'y']
        self.alphabet_len = len(self.vn_alphabet)
        self.char_couples = [['i', 'y'], ['s', 'x'], ['gi', 'd'],
                ['ă', 'â'], ['ch', 'tr'], ['ng', 'n'], 
                ['nh', 'n'], ['ngh', 'ng'], ['ục', 'uộc'], ['o', 'u'], 
                ['ă', 'a'], ['o', 'ô'], ['ả', 'ã'], ['ổ', 'ỗ'], ['ủ', 'ũ'], ['ễ', 'ể'], 
                ['e', 'ê'], ['à', 'ờ'], ['ằ', 'à'], ['ẩn', 'uẩn'],  ['ẽ', 'ẻ'], ['ùi', 'uồi'], ['ă', 'â'], ['ở', 'ỡ'], ['ỹ', 'ỷ'], ['ỉ', 'ĩ'], ['ị', 'ỵ'],
                ['ấ', 'á'],['n', 'l'], ['qu', 'w'], ['ph', 'f'], ['d', 'z'], ['c', 'k'], ['qu', 'q'], ['i','j'], ['gi', 'j'], 
                ]

        self.teencode_dict = {'mình': ['mk', 'mik', 'mjk'], 'vô': ['zô', 'zo', 'vo'], 'vậy':['zậy', 'z', 'zay', 'za'] , 'phải': ['fải', 'fai', ], 'biết': ['bit', 'biet'], 
                              'rồi':['rùi', 'ròi', 'r'], 'bây': ['bi', 'bay'], 'giờ': ['h', ], 'không': ['k', 'ko', 'khong', 'hk', 'hong', 'hông', '0', 'kg', 'kh', ], 
                              'đi': ['di', 'dj', ], 'gì': ['j', ], 'em': ['e', ], 'được': ['dc', 'đc', ], 'tao': ['t'], 'tôi': ['t'], 'chồng': ['ck'], 'vợ':['vk']

        }

        self.all_word_candidates = self.get_all_word_candidates(self.word_couples)
        self.string_all_word_candidates = ' '.join(self.all_word_candidates)
        self.all_char_candidates = self.get_all_char_candidates( )

    def replace_teencode(self, word):
        candidates = self.teencode_dict.get(word, None)
        if candidates is not None:
            chosen_one = 0
            if len(candidates) > 1:
                chosen_one = np.random.randint(0, len(candidates))
            return candidates[chosen_one]
         
    def replace_word_candidate(self, word):
        """
        Return a homophone word of the input word.
        """
        capital_flag = word[0].isupper()
        word = word.lower()
        if capital_flag and word in self.teencode_dict:
            return self.replace_teencode(word).capitalize()
        elif word in self.teencode_dict:
            return self.replace_teencode(word)

        for couple in self.word_couples:
            for i in range(2):
                if couple[i] == word:
                    if i == 0:
                        if capital_flag:
                            return couple[1].capitalize()
                        else:
                            return couple[1]
                    else:
                        if capital_flag:
                            return couple[0].capitalize()
                        else:
                            return couple[0]

    def replace_char_candidate(self,char):
        """
        return a homophone char/subword of the input char.
        """
        for couple in self.char_couples:
            for i in range(2):
                if couple[i] == char:
                    if i == 0:
                        return couple[1]
                    else:
                        return couple[0]

    def get_all_char_candidates(self, ):
        
        all_char_candidates = []
        for couple in self.char_couples:
            all_char_candidates.extend(couple)
        return all_char_candidates


    def get_all_word_candidates(self, word_couples):

        all_word_candidates = []
        for couple in self.word_couples:
            all_word_candidates.extend(couple)
        return all_word_candidates

    def remove_diacritics(self, text, onehot_label):
        """
        Replace word which has diacritics with the same word without diacritics
        Args: 
            text: a list of word tokens
            onehot_label: onehot array indicate position of word that has already modify, so this 
            function only choose the word that do not has onehot label == 1.
        return: a list of word tokens has one word that its diacritics was removed, 
                a list of onehot label indicate the position of words that has been modified.
        """
        idx = np.random.randint(0, len(onehot_label))
        prevent_loop = 0
        while onehot_label[idx] == 1 or text[idx] == unidecode.unidecode(text[idx]) or text[idx] in string.punctuation:
            idx = np.random.randint(0, len(onehot_label))
            prevent_loop += 1
            if prevent_loop > 10:
                return False, text, onehot_label


        onehot_label[idx] = 1
        text[idx] = unidecode.unidecode(text[idx])
        return True, text, onehot_label

    def replace_with_random_letter(self, text, onehot_label):
        """
        Replace, add (or remove) a random letter in a random chosen word with a random letter
        Args:
            text: a list of word tokens
            onehot_label: onehot array indicate position of word that has already modify, so this 
            function only choose the word that do not has onehot label == 1. 
        return: a list of word tokens has one word that has been modified, 
                a list of onehot label indicate the position of words that has been modified.
        """
        idx = np.random.randint(0, len(onehot_label))
        prevent_loop = 0
        while onehot_label[idx] == 1 or text[idx].isnumeric() or text[idx] in string.punctuation :
            idx = np.random.randint(0, len(onehot_label))
            prevent_loop += 1
            if prevent_loop > 10:
                return False, text, onehot_label

        # replace, add or remove? 0 is replace, 1 is add, 2 is remove
        coin = np.random.choice([0, 1,2])
        if coin == 0:
            chosen_letter = text[idx][np.random.randint(0, len(text[idx]))]
            replaced = self.vn_alphabet[np.random.randint(0, self.alphabet_len)]
            try:
                text[idx] = re.sub(chosen_letter,replaced , text[idx])
            except:
                return False, text, onehot_label
        elif coin == 1:
            chosen_letter = text[idx][np.random.randint(0, len(text[idx]))]
            replaced = chosen_letter + self.vn_alphabet[np.random.randint(0, self.alphabet_len)]
            try:
                text[idx] = re.sub(chosen_letter,replaced , text[idx])
            except:
                return False, text, onehot_label
        else:
            chosen_letter = text[idx][np.random.randint(0, len(text[idx]))]
            try:
                text[idx] = re.sub(chosen_letter, '', text[idx])  
            except:
                return False, text, onehot_label   

        onehot_label[idx] = 1
        return True, text, onehot_label

    def replace_with_homophone_word(self, text, onehot_label):
        """
        Replace a candidate word (if exist in the word_couple) with its homophone. if successful, return True, else False
        Args:
            text: a list of word tokens
            onehot_label: onehot array indicate position of word that has already modify, so this 
            function only choose the word that do not has onehot label == 1. 
        return: True, text, onehot_label if successful replace, else False, text, onehot_label
        """
        # account for the case that the word in the text is upper case but its lowercase match the candidates list
        candidates = []
        for i in range(len(text)):
            if text[i].lower() in self.all_word_candidates or text[i].lower() in self.teencode_dict.keys():
                candidates.append((i, text[i]))
        
        if len(candidates) == 0:
            return False, text, onehot_label

        idx = np.random.randint(0, len(candidates))
        prevent_loop = 0
        while onehot_label[candidates[idx][0]] == 1:
            idx = np.random.choice(np.arange(0, len(candidates)))
            prevent_loop += 1
            if prevent_loop > 5:
                return False, text, onehot_label

        text[candidates[idx][0]] = self.replace_word_candidate(candidates[idx][1])
        onehot_label[candidates[idx][0]] = 1
        return True, text, onehot_label

    def replace_with_homophone_letter(self, text, onehot_label):
        """
        Replace a subword/letter with its homophones
        Args:
            text: a list of word tokens
            onehot_label: onehot array indicate position of word that has already modify, so this 
            function only choose the word that do not has onehot label == 1. 
        return: True, text, onehot_label if successful replace, else False, None, None
        """
        candidates = []
        for i in range(len(text)):
            for char in self.all_char_candidates:
                if re.search(char, text[i]) is not None:
                    candidates.append((i, char))
                    break

        if len(candidates) == 0:

           return False, text, onehot_label
        else:
            idx = np.random.randint(0, len(candidates))
            prevent_loop = 0
            while onehot_label[candidates[idx][0]] == 1:
                idx = np.random.randint(0, len(candidates))
                prevent_loop += 1
                if prevent_loop  > 5:
                    return False, text, onehot_label

            replaced = self.replace_char_candidate(candidates[idx][1])
            text[candidates[idx][0]] = re.sub(candidates[idx][1], replaced, text[candidates[idx][0]] )

            onehot_label[candidates[idx][0]] = 1
            return True, text, onehot_label


    def replace_with_random_word(self,text, onehot_label):
        """
        Replace a random word in text with a random word in vocab.
        Args:
            text: a list of word tokens
            onehot_label: onehot array indicate position of word that has already modify, so this 
            function only choose the word that do not has onehot label == 1. 
        return: True, text, onehot_label if successful replace, else False, text, onehot_label

        """
        idx = np.random.randint(0, len(text))
        prevent_loop  = 0
        # the idx must not be an already modify token, punctuation or number.
        while onehot_label[idx] == 1 or text[idx].isnumeric() or text[idx] in string.punctuation:
            idx = np.random.randint(0, len(text))
            prevent_loop += 1
            if prevent_loop > 10:
                return False, text, onehot_label

        chosen_idx = np.random.randint(0, len(self.vocab))
        text[idx] = self.vocab[chosen_idx]

        onehot_label[idx] = 1
        return True, text, onehot_label


    def create_wrong_word(self, text, mode):
        """
        Function to create miss-spelled words and its label.
        Args: 
            text: One sentence of text.
            mode: which type of error to create, can be: random_word, homophone_char,
            random_letter, remove_diacritics, 
                                                        
        return: raw_text, onehot_label, text_label
        """
        text = self.tokenizer(text)
        text_label = text.copy()
        onehot_label = [0]*len(text)
        num_wrong = int(np.round(0.15*len(text)))
        
        if mode == 'random_word':
            for i in range(0, num_wrong):
                _, text, onehot_label = self.replace_with_random_word(text, onehot_label) 
                if not _:
                    #logger.info('False to create wrong word with random word!')
                    return False, (text, onehot_label, text_label)

                    
            return True, (text, onehot_label, text_label)

        elif mode == 'homophone_char':
            for i in range(0, num_wrong):
                _, text, onehot_label = self.replace_with_homophone_letter(text, onehot_label)
                if not _:
                    #logger.info('False to create wrong word with homophone_char')  
                    return False, (onehot_label, text_label)   
            return True, (text, onehot_label, text_label)    

        elif mode == 'homophone_word':
            for i in range(0, num_wrong):
                _, text, onehot_label = self.replace_with_homophone_word(text, onehot_label)
                if not _:
                    #logger.info('False to create wrong word with homophone_word, remove with random_word')
                    #_, text, onehot_label = self.replace_with_random_word(text, onehot_label) 
                    return False, (text, onehot_label, text_label)
            return True, (text, onehot_label, text_label)

        elif mode == 'random_letter':
            for i in range(0, num_wrong):
                _, text, onehot_label = self.replace_with_random_letter(text, onehot_label)
                if not _:
                    #logger.info('False to create wrong word with random_letter, remove with random_word') 
                    return False, (text, onehot_label, text_label) 
            return True, (text, onehot_label, text_label)         

        elif mode == 'remove_diacritics':
            for i in range(0 ,num_wrong):
                _, text, onehot_label = self.remove_diacritics(text, onehot_label)
                if not _:
                    #logger.info('False to create wrong word with remove_diacritics, remove with random_word')
                    return False, (text, onehot_label, text_label)
            return True, (text, onehot_label, text_label)

if __name__ == 'main':
    synthesizer = SynthesizeData(vocab_path = vocab_path)
    random_word_data = []
    for line in tqdm(all_sentences):
        _, r = synthesizer.create_wrong_word(line, mode = 'random_word')
        if _:
            random_word_data.append(r)
    data_path = '/content/drive/MyDrive/nlp_projects/Text_correction/all_data/random_word_data'
    write_data(data_path, random_word_data)