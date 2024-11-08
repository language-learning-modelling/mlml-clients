import os
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

def write_batch_file(
                output_fp,
                data_dict,
                     ):
    with open(output_fp,"w") as batch_outf:
        dict_str = json.dumps(
                data_dict,
                indent=4)
        batch_outf.write(dict_str)

def flag_already_processed_for_given_model(
        texts_dict_dict,
        model_name 
    ):
    filtered_data = texts_dict_dict.copy()
    for text_id, text_dict in list(texts_dict_dict.items())[::-1]:
     f=all(token_dict["predictions"]["models"].get(model_name, False)
        for token_idx, token_dict in enumerate(text_dict["tokens"]))
     if f:
         del filtered_data[text_id] 
         '''filtered_data[text_id] = {
                 f"{model_name}_is_processed": True,
                 }'''
     elif not f:
         pass
         '''filtered_data[text_id] = {
                 f"{model_name}_is_processed": False
                 }'''
     # filtered_data[text_id].update(text_dict) 
    return filtered_data

def load_input_or_partial(input_fp, output_folder):
    expected_partial=f"{config.OUTPUT_FOLDER}/partial/{config.INPUT_FILENAME}_{config.MODEL_NAME}.json.zlib"
    if os.path.exists(expected_partial):
        texts = json.load(open(expected_partial))
    else:
        texts = json.load(open(input_fp))
    return texts

if __name__ == "__main__":
    config_fp_or_jsonstr = "".join(sys.argv[1:])
    config_dict = load_config(config_fp_or_jsonstr)
    config = Config(**config_dict) 
    config.INPUT_FILENAME = config.INPUT_FP.split("/")[-1] 
    config.MODEL_NAME = config.MODEL_CHECKPOINT.split("/")[-1] 
    writing_batch = load_input_or_partial(
                config.INPUT_FP,
                config.OUTPUT_FOLDER
                ) 
    print(f' original input file has {len(writing_batch.keys())} texts')
    config.TEXTS = flag_already_processed_for_given_model(
            writing_batch,
            config.MODEL_NAME 
            )
    print(f' after flagging already processed for {config.MODEL_NAME} texts has {len(config.TEXTS.keys())} texts to be processed')
    #import random
    #sample_keys = random.sample(sorted(config.TEXTS.keys()),30) 
    #config.TEXTS = {k:config.TEXTS[k] for k in sample_keys} 
    p = Predictor(config_obj=config)
    n_of_maskedsentences = sum(len(text_d['tokens']) for text_d in config.TEXTS.values())
    n_of_saving_steps = 10 
    writing_size = n_of_maskedsentences // n_of_saving_steps\
            if   (n_of_maskedsentences % n_of_saving_steps) == 0\
            else (n_of_maskedsentences // n_of_saving_steps) + 1
    n_of_iterations = n_of_maskedsentences // config.BATCH_SIZE\
            if   (n_of_maskedsentences % config.BATCH_SIZE) == 0\
            else (n_of_maskedsentences // config.BATCH_SIZE) + 1
    pbar = tqdm(range(n_of_iterations))
    processed_count=0
    batch_generator = p.predict()
    for batch_idx in pbar:
        s=time.time()
        try:
            ranked_vocab_dict_per_masked_sentence = next(batch_generator)
        except StopIteration:
            break

        processed_count+=len(ranked_vocab_dict_per_masked_sentence)
        for mlm_id, preds_dict_lst\
                in ranked_vocab_dict_per_masked_sentence.items():

            text_id, token_idx=mlm_id.split("_")[-2:]
            token_idx = int(token_idx)
            writing_batch[text_id]["tokens"][token_idx]["predictions"]["models"][config.MODEL_NAME] = preds_dict_lst
        # writing_batch.update(ranked_vocab_dict_per_masked_sentence)
        if processed_count >= writing_size:
            writing_size+=writing_size
            batch_outfp=f"{config.OUTPUT_FOLDER}/partial/{config.INPUT_FILENAME}_{config.MODEL_NAME}.json"
            write_batch_file(
                    batch_outfp,
                    writing_batch
            )
        elapsed=time.time()-s
        pbar.set_description(f'# proc : {len(ranked_vocab_dict_per_masked_sentence)} total : {processed_count} save when reaches: {writing_size} it {elapsed} seconds')
    batch_outfp=f"{config.OUTPUT_FOLDER}/{config.INPUT_FILENAME}_{config.MODEL_NAME}.json"
    write_batch_file(
            batch_outfp,
            writing_batch
    )
