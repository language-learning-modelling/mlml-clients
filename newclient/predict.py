import sys
from mlml_hugginface import Predictor

config_fp = sys.argv[1]

p = Predictor(config_filepath=config_fp)
p.predict()
