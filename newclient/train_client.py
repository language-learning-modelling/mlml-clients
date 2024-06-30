from mlml_hugginface import Trainer, Downloader
import sys

training_fp = sys.argv[1]

downloader = Downloader("./run_configs/download_xlm_roberta_base.json")
downloader.downloadLocally()

trainer = Trainer(training_fp)
trainer.train()
