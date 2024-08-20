from mlml_hugginface import Downloader
import sys

download_config_fp = sys.argv[1]
downloader = Downloader(download_config_fp)
downloader.downloadLocally()
