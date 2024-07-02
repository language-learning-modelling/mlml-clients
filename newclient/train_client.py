from mlml_hugginface import Trainer, Downloader
import sys

training_config_fp = sys.argv[1]

if len(sys.argv) > 2:
    download_config_fp = sys.argv[2]
    downloader = Downloader(download_config_fp)
    downloader.downloadLocally()

'''
trainer = Trainer(training_config_fp)
trainer.train()
'''
