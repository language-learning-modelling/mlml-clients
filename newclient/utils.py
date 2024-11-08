import json
import os
import zlib

def load_maskedsentence_txt(filepath, filename):
    data = {}
    with open(filepath) as inpf:
        for line_idx, line in enumerate(inpf):
            line=line.replace("\n","")
            data[f'{filename}_{line_idx}'] = line 
    return data

def load_config(config_fp_or_jsonstr):
    if os.path.exists(config_fp_or_jsonstr): 
        with open(config_fp_or_jsonstr) as inpf:
            config = json.load(inpf)
            config = {k.upper(): v for k, v in config.items()}
            return config
    else:
        return { k.upper():v for (k,v) in json.loads(config_fp_or_jsonstr).items() } 

def compress_dict(data):
    # Convert the dictionary to a JSON string
    json_string = json.dumps(data)

    # Encode the JSON string to bytes
    json_bytes = json_string.encode('utf-8')

    # Compress the byte data using zlib
    compressed_data = zlib.compress(json_bytes)

    return compressed_data

def decompress_dict(compressed_data):
    # Decompress the data using zlib
    decompressed_bytes = zlib.decompress(compressed_data)

    # Decode the bytes back to a JSON string
    json_string = decompressed_bytes.decode('utf-8')

    # Convert the JSON string back to a dictionary
    data = json.loads(json_string)

    return data
