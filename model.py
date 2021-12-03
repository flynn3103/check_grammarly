class Detector(nn.Module):
    def __init__(self, input_dim,output_dim,  embedding_dim, num_layers, hidden_size):

        super(Detector, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_dim  = embedding_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings = self.input_dim, embedding_dim = self.embedding_dim, )
        self.LSTM = nn.LSTM(input_size = self.embedding_dim, hidden_size= self.hidden_size, num_layers = self.num_layers, 
                            batch_first = True, dropout = 0.1, bidirectional = True)
        self.linear = nn.Linear(self.hidden_size*2, self.output_dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        emb = self.embedding(x)
        outputs, (h_n, h_c) = self.LSTM(emb)
        logits = self.linear(outputs)
        p = self.sigmoid(logits)
        return p

class HardMasked(nn.Module):
    def __init__(self, detector, MaskedLM, detector_tokenizer, maskedlm_tokenzier,device ):
        super(HardMasked, self).__init__()

        self.detector = detector.to(device)
        self.MaskedLM = MaskedLM.to(device)
        self.detector_tokenizer = detector_tokenizer
        self.maskedlm_tokenizer = maskedlm_tokenizer
        self.use_device = device


    def forward(self, s):
        maskedlm_features = self.prepare_input(s)
        outputs = MaskedLM(input_ids = torch.tensor([maskedlm_features['input_ids']], dtype = torch.long, device = self.use_device), 
                            attention_mask = torch.tensor([maskedlm_features['attention_mask']], dtype = torch.long, device = self.use_device) )
        logits = outputs['logits'][0]
        output_ids = torch.argmax(logits, dim = -1)
        final_output = maskedlm_tokenizer.decode(output_ids)
        return final_output
        
    def prepare_input(self, s):

        detector_input_ids = self.detector_tokenizer.encode(s, out_type = int)
        detector_input_pieces = self.detector_tokenizer.id_to_piece(detector_input_ids)
        detector_outputs = (self.detector(torch.tensor([detector_input_ids], dtype = torch.long, device = self.use_device))[0].reshape(1,-1) > 0.5).int()[0] 

        for i in range(1, len(detector_input_pieces)):
            if detector_outputs[i] == 1:
                detector_input_pieces[i] = ' <mask>'

        masked_s = self.detector_tokenizer.decode(detector_input_pieces)
        for i in range(5):
            masked_s = re.sub(r'<mask>\s<mask>', '<mask>', masked_s)

        maskedlm_features = maskedlm_tokenizer(masked_s)

        return maskedlm_features