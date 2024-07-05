import sys
import json
from mlml_hugginface import Predictor
from utils import load_config
from tqdm import tqdm
from dataclasses import dataclass
from utils import load_config, load_maskedsentence_txt
import time

@dataclass
class Config:
    INPUT_FP: str = None
    OUTPUT_FOLDER: str = None
    MODEL_CHECKPOINT: str = None
    BATCH_SIZE: str = None
    TOP_K: str = None

    def __post_init__(self):
        for field_key in self.__dataclass_fields__.keys():
            if self.__getattribute__(field_key) is None:
             raise ValueError(f'missing {field_key} config property')

def write_batch_file(OUTPUT_FOLDER,
                INPUT_FILENAME,
                MODEL_CHECKPOINT,
                batch_idx,
                data_dict,
                is_batch=True
                     ):
    if is_batch:
        batch_outfp=f"{OUTPUT_FOLDER}/{INPUT_FILENAME}_batch_{batch_idx}_{MODEL_CHECKPOINT}.json"
    else:
        batch_outfp=f"{OUTPUT_FOLDER}/{INPUT_FILENAME}_{MODEL_CHECKPOINT}.json"
    with open(batch_outfp,"w") as batch_outf:
        dict_str = json.dumps(
                data_dict,
                indent=4)
        batch_outf.write(dict_str)

def filter_already_processed_for_given_model(
        texts_dict_dict,
        model_name 
    ):
    print(len(texts_dict_dict.keys()))
    filtered_data = texts_dict_dict.copy()
    for text_id, text_dict in texts_dict_dict.items():
     f=all(token_dict["predictions"]["models"].get(model_name, False)
        for token_idx, token_dict in enumerate(text_dict["tokens"]))
     if f:
         del filtered_data[text_id]
    return filtered_data
def check_each_text_that_has_a_prediction_has_for_all_tokens(text_dict):
    pass
if __name__ == "__main__":
    config_fp_or_jsonstr = "".join(sys.argv[1:])
    config_dict = load_config(config_fp_or_jsonstr)
    config = Config(**config_dict) 
    config.INPUT_FILENAME = config.INPUT_FP.split("/")[-1] 
    config.MODEL_NAME = config.MODEL_CHECKPOINT.split("/")[-1] 
    config.TEXTS = filter_already_processed_for_given_model(
            json.load(open(config.INPUT_FP)), 
            config.MODEL_NAME 
            )
    #import random
    #sample_keys = random.sample(sorted(config.TEXTS.keys()),30) 
    #config.TEXTS = {k:config.TEXTS[k] for k in sample_keys} 
    p = Predictor(config_obj=config)
    writing_batch = config.TEXTS.copy() 
    writing_size = 500
    n_of_maskedsentences = sum(len(text_d['tokens']) for text_d in config.TEXTS.values())
    n_of_iterations = n_of_maskedsentences // config.BATCH_SIZE\
            if   (n_of_maskedsentences % config.BATCH_SIZE) == 0\
            else (n_of_maskedsentences // config.BATCH_SIZE) + 1
    pbar = tqdm(range(n_of_iterations))
    processed_count=0
    batch_generator = p.predict()
    for batch_idx in pbar:
        s=time.time()
        ranked_vocab_dict_per_masked_sentence = next(batch_generator)
        processed_count+=len(ranked_vocab_dict_per_masked_sentence)
        print(f'# of MS processed : {len(ranked_vocab_dict_per_masked_sentence)} totalling : {processed_count}')
        for mlm_id, preds_dict_lst\
                in ranked_vocab_dict_per_masked_sentence.items():

            text_id, token_idx=mlm_id.split("_")[-2:]
            token_idx = int(token_idx)
            writing_batch[text_id]["tokens"][token_idx]["predictions"]["models"][config.MODEL_NAME] = preds_dict_lst
        elapsed=time.time()-s
        pbar.set_description(f"iteration took {elapsed} seconds")
        # writing_batch.update(ranked_vocab_dict_per_masked_sentence)
        '''
        if len(writing_batch) >= writing_size:
            write_batch_file(
                    config.OUTPUT_FOLDER,
                    config.INPUT_FILENAME, 
                    config.MODEL_NAME,
                    batch_idx,
                    writing_batch
            )
            writing_batch = {}
        '''
    write_batch_file(
            config.OUTPUT_FOLDER,
            config.INPUT_FILENAME, 
            config.MODEL_NAME,
            batch_idx,
            writing_batch,
            is_batch=False
    )
