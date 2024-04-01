from mlml_hugginface import Trainer, Downloader

downloader = Downloader("./run_configs/download_distillbert.json")
downloader.downloadLocally()

'''
trainer = Trainer("./run_configs/train.json")
trainer.train()
'''

