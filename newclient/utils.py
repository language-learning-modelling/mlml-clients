import json
import os

def load_maskedsentence_txt(filepath, filename):
    data = {}
    with open(filepath) as inpf:
        for line_idx, line in enumerate(inpf):
            line=line.replace("\n","")
            data[f'{filename}_{line_idx}'] = line 
    return data

def load_config(config_fp_or_jsonstr):
    if os.path.exists(config_fp_or_jsonstr): 
        with open(config_filepath_or_dictstr) as inpf:
            config = json.load(inpf)
            config = {k.upper(): v for k, v in config.items()}
            return config
    else:
        return { k.upper():v for (k,v) in json.loads(config_fp_or_jsonstr).items() } 
