import numpy as np

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