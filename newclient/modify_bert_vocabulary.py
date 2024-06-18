from mlml_hugginface import add_tokens_to_bert_vocabulary 
import sys


config = {
        "prefix_tokens_fp": "./datasets/EFCAMDAT/prefix_tokens.txt",
        "outputfp": "./models/efcamdat-prefix-tokens-vocab-bert-base-uncased"
    }

with open(config["prefix_tokens_fp"]) as prefix_inpf:
    prefix_tokens=[prefix_token.replace("\n","") for prefix_token in prefix_inpf]

print(prefix_tokens)
add_tokens_to_bert_vocabulary(
        new_tokens_str=prefix_tokens,
        outputfp=config["outputfp"],
        )
