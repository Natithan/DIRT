import sys
from datetime import datetime

from absl import flags
#TODO change dropout prob to 0.1 in line with t5
# %% FLAGS
FLAGS = flags.FLAGS
flags.DEFINE_integer("d_batch", 10, "Batch size")
flags.DEFINE_integer("d_emb", 12, "Size of token encodings before contextualization")
flags.DEFINE_integer("d_hidden", 72, "Size of token encodings in hidden layers (contextualized)")
flags.DEFINE_integer("nb_heads", 8, "Number of attention heads")
flags.DEFINE_integer("device_idx", 1, "GPU index. -1 for CPU")
# flags.DEFINE_integer("target_length", 20, "Number of tokens in target sequence")
flags.DEFINE_float("masking_fraction", .15, "Fraction of tokens to be masked during MLM pretraining")
# flags.DEFINE_integer("source_length", 20, "Number of tokens in source sequence")
flags.DEFINE_integer("max_seq_length", 40, "Maximum number of words to consider per batch")
flags.DEFINE_string("data_folder", "./data/Gutenberg", "Folder with train, val and test subfolders containing data")
flags.DEFINE_string("model_folder", "./output", "Folder with trained models and tensorboard logs")
flags.DEFINE_string("run_name", datetime.now().strftime("%b_%d_%Hh%Mm%Ss"), "Folder with trained models and tensorboard logs")
flags.DEFINE_string("mode","", "Flag to allow python console command line argument")
# Trainer flags
flags.DEFINE_integer("patience", 1000, "Number of epochs the validation metric can worsen before stopping training.")
flags.DEFINE_integer("num_epochs", 1000, "Number of epochs to train for.")

flags.DEFINE_bool("mini", True, "Whether to work with mini data/models for debugging purposes")

flags.DEFINE_integer("nb_encoder_layers", 6, "Number of layers in the encoder.")
flags.DEFINE_integer("nb_decoder_layers", 6, "Number of layers in the decoder.")
flags.DEFINE_integer("nb_feedforward_layers", 2, "Number of layers in the feedforward subcomponents of the transformer.")
FLAGS(sys.argv)