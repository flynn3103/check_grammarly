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

def get_data(data_path, num_train, num_dev):
    raw_text = []
    onehot_label = []
    text_label = []
    with open(os.path.join(data_path, 'rawtext.txt'), 'r', encoding = 'utf-8') as f:
        for line in f:
            raw_text.append(line.rstrip())
    with open(os.path.join(data_path, 'onehot_label.txt'), 'r') as f:
        for line in f:
            onehot_label.append([int(x) for x in  line.rstrip().split()])

    with open(os.path.join(data_path, 'text_label.txt'), 'r', encoding = 'utf-8') as f:
        for line in f:
            text_label.append(line.rstrip())

    assert len(raw_text) == len(onehot_label) == len(text_label), 'Error: len data does not match'
    total = num_train + num_dev
    indices = np.random.randint(0, len(onehot_label), total)
    train_data  = []
    dev_data = []
    for idx in indices[:num_train]:
        train_data.append((raw_text[idx], onehot_label[idx], text_label[idx]))
    for idx in indices[num_train:]:
        dev_data.append((raw_text[idx], onehot_label[idx], text_label[idx]))
    
    return train_data, dev_data