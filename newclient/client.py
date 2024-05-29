from mlml_hugginface import Trainer, Downloader
import sys

training_fp = sys.argv[1]

downloader = Downloader("./run_configs/download_bert_base_uncased.json")
downloader.downloadLocally()

trainer = Trainer(training_fp)
trainer.train()
