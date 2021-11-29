import numpy as np
import logging

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def _read_file(input_file, min_seq_len, max_seq_len):
  with open(input_file, 'r', encoding= 'utf-8') as f:
    raw_text= f.read()

  lines = []
  raw_text = raw_text.split()
  raw_text_len = len(raw_text)
  while True:
    line_len = np.random.randint(min_seq_len, max_seq_len)
    if raw_text_len < line_len :
      break
    line = ' '.join(raw_text[:line_len])
    raw_text = raw_text[line_len:]
    lines.append(line)
    raw_text_len -= line_len
    
  return lines

def write_data(data_path, data):
    tree = TreebankWordDetokenizer()
    with open(os.path.join(data_path, 'rawtext.txt'), 'w', encoding = 'utf-8') as f:
        for line in tqdm(data, desc = 'Writing rawtext file...'):
            f.write(tree.detokenize(line[0]) + '\n')
    with open(os.path.join(data_path, 'onehot_label.txt'), 'w', encoding = 'utf-8') as f:
        for line in tqdm(data, desc = 'Writing onehot_label file...'):
            f.write(' '.join([str(x) for x in line[1]]) + '\n')
    with open(os.path.join(data_path, 'text_label.txt'), 'w', encoding = 'utf-8') as f:
        for line in tqdm(data, desc = 'Writing text_label file...'):
            f.write(tree.detokenize(line[2]) + '\n')

def num_parameters(parameters):
    num = 0
    for i in parameters:
        num += len(i)
    return num

