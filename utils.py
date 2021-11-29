import numpy as np


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

    
def num_parameters(parameters):
    num = 0
    for i in parameters:
        num += len(i)
    return num