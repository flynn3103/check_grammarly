detector_path = '/content/Detector967.pkl'
detector_tokenizer_path = '/content/spm_tokenizer.model'

MaskedLM = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-base')

maskedlm_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

detector_tokenizer = spm.SentencePieceProcessor(detector_tokenizer_path, )

detector = torch.load(detector_path)


model = HardMasked(detector, MaskedLM, detector_tokenizer, maskedlm_tokenizer, 'cuda')

s = 'Tôi vẫn luôn iu cô ấy với hết tấm lòng của mk'
model(s)