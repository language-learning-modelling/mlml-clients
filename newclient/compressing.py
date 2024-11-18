if __name__ == "__main__":
    import json
    from utils import compress_dict, decompress_dict
    #Sample dictionary
    #my_dict = {
    #   "name": "Alice",
    #   "age": 30,
    #   "city": "Wonderland",
    #   "hobbies": ["reading", "adventures", "tea parties"]
    #}
    #with open("./datasets/EFCAMDAT/cleaned_efcamdat.json") as inpf:
    #   my_dict = json.load(inpf)

    # Compress the dictionary
    # compressed = compress_dict(my_dict)
    filepath= "./datasets/CELVA/predictions_batch/partial/tokenized_celva.csv.json.zlib_bert-base-uncased.json.zlib"
    #with open(filepath,"wb") as outf:
    #    outf.write(compressed)
    def read_dataset_zlib_json(filepath):
        with open(filepath, "rb") as inpf:
            decompressed = decompress_dict(inpf.read())
        return decompressed
    data = read_dataset_zlib_json(filepath)
    print(data)
    #print("Compressed data:", compressed)
    # Decompress the dictionary
    # decompressed = decompress_dict(compressed)

