from pathlib import Path
from os import path

#Change each experiment
NAME = 'V.0.0-Beta'

NUM_EPOCHS = 1

SEED = 42

BATCH_SIZE = 32

############################### DIRECTORIES ##############################

#
MAIN_DIR = Path(path.abspath(__file__)).parent.parent.parent
##
DATASET_DIR = MAIN_DIR.joinpath('data')
##
END_MODEL_DIR = MAIN_DIR.joinpath('end_model')
##
TRAIN_DIR = END_MODEL_DIR.joinpath('training')
##
SERIALIZATION_DIR = END_MODEL_DIR.joinpath('serialization')

########################################################################
