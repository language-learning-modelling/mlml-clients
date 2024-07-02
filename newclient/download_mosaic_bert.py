import torch
import transformers
from transformers import AutoModelForMaskedLM, BertTokenizer, pipeline
from transformers import BertTokenizer, BertConfig

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # MosaicBERT uses the standard BERT tokenizer

config = transformers.BertConfig.from_pretrained('mosaicml/mosaic-bert-base') # the config needs to be passed in
mosaicbert = AutoModelForMaskedLM.from_pretrained('mosaicml/mosaic-bert-base',config=config,trust_remote_code=True)

