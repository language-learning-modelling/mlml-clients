import sys
import json
from mlml_hugginface import Predictor
from utils import load_config
from tqdm import tqdm
from dataclasses import dataclass
from utils import load_config, load_maskedsentence_txt

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
                batch_idx,
                writing_batch):
    batch_outfp=f"{OUTPUT_FOLDER}/{INPUT_FILENAME}_batch_{batch_idx}.json"
    with open(batch_outfp,"w") as batch_outf:
        dict_str = json.dumps(
                writing_batch,
                indent=4)
        batch_outf.write(dict_str)

if __name__ == "__main__":
    config_fp_or_jsonstr = "".join(sys.argv[1:])
    config_dict = load_config(config_fp_or_jsonstr)
    config = Config(**config_dict) 
    config.INPUT_FILENAME = config.INPUT_FP.split("/")[-1] 
    config.TEXTS = load_maskedsentence_txt(
            config.INPUT_FP,config.INPUT_FILENAME 
            )
    p = Predictor(config_dict=config)
    writing_batch = {}
    writing_size = 500
    for batch_idx, ranked_vocab_dict_per_masked_sentence \
            in enumerate(tqdm(p.predict())):
        print(ranked_vocab_dict_per_masked_sentence.keys())
        writing_batch.update(ranked_vocab_dict_per_masked_sentence)
        if len(writing_batch) >= writing_size:
            write_batch_file(
                    config.OUTPUT_FOLDER,
                    config.INPUT_FILENAME, 
                    batch_idx,
                    writing_batch
            )
            writing_batch = {}
    if len(writing_batch) >= 0:
            write_batch_file(
                    config.OUTPUT_FP_TEMPLATE,
                    config.INPUT_FILENAME, 
                    batch_idx,
                    writing_batch
            )
