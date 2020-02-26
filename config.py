import logging
from collections import OrderedDict

import sys
from datetime import datetime
from absl import flags
# %% FLAGS
from util import get_freer_gpu
logger = logging.getLogger(__name__)


# TODO maybe make multiple configs? Or maybe keep model hyperparams in some config, and use FLAGS just for folder names etc
FLAGS = flags.FLAGS
flags.DEFINE_integer("d_batch", 20, "Batch size")
flags.DEFINE_string("model", "my_baseline_encoder", "Name of the model to use (see MODEL_MAPPING)")
flags.DEFINE_string("objective", "simple_mlm",
                    "Name of the denoising objective to use (see OBJECTIVE_MAPPING)")
flags.DEFINE_integer("d_emb", 72, "Size of token encodings before contextualization")
flags.DEFINE_integer("d_hidden", 768, "Size of token encodings in hidden layers (contextualized)")
flags.DEFINE_integer("d_ff", 3072, "Number of hidden units in feedforward parts of attention blocks")
flags.DEFINE_integer("nb_heads", 8, "Number of attention heads")
flags.DEFINE_integer("device_idx", get_freer_gpu(), "GPU index. -1 for CPU. Defaults to the GPU with most free memory")
flags.DEFINE_float("masking_fraction", .15, "Fraction of tokens to be masked during MLM pretraining")
flags.DEFINE_float("dropout_rate", .1, "Dropout rate")
flags.DEFINE_integer("max_seq_length", 512, "Maximum number of words to consider per batch")
flags.DEFINE_string("data_folder", "./data/Gutenberg", "Folder with train, val and test subfolders containing data")
flags.DEFINE_string("model_folder", "./output", "Folder with trained models and tensorboard logs")
flags.DEFINE_string("run_name", datetime.now().strftime("%b_%d_%Hh%Mm%Ss"),
                    "Folder with trained models and tensorboard logs")
flags.DEFINE_string("mode", "", "Flag to allow python console command line argument")
# Trainer flags
flags.DEFINE_integer("patience", 500, "Number of epochs the validation metric can worsen before stopping training.")
flags.DEFINE_integer("num_epochs", 10000, "Number of epochs to train for.")

flags.DEFINE_bool("mini", False, "Whether to work with mini data/models for debugging purposes")

flags.DEFINE_bool("use_decoder", True, "Whether to use a Transformer decoder on top of the encoder")

flags.DEFINE_integer("nb_encoder_layers", 6, "Number of layers in the encoder.")
flags.DEFINE_integer("nb_decoder_layers", 6, "Number of layers in the decoder.")
flags.DEFINE_integer("nb_feedforward_layers", 2,
                     "Number of layers in the feedforward subcomponents of the transformer.")
flags.DEFINE_integer("relative_attention_num_buckets", 32, "Number of different position embeddings.")
flags.DEFINE_integer("beam_width", 3, "Width of the beam during the decoding beam search phase.")
flags.DEFINE_integer("max_vocab_size", 30000, "Maximum number of different tokens to differentiate.")

FLAGS(sys.argv)
# class BaseConfig(object):
#     """
#     Base class for configuration files
#     """
#     def __init__(self, **kwargs):
#         self.d_batch = kwargs.pop("d_batch", 20)
#         self.output_attentions = kwargs.pop("output_attentions", False)
#         self.output_attentions = kwargs.pop("output_attentions", False)
#         self.output_attentions = kwargs.pop("output_attentions", False)
#         self.output_attentions = kwargs.pop("output_attentions", False)
#         self.output_attentions = kwargs.pop("output_attentions", False)
#         self.output_attentions = kwargs.pop("output_attentions", False)
#         self.output_attentions = kwargs.pop("output_attentions", False)
#         self.output_attentions = kwargs.pop("output_attentions", False)
#         self.output_attentions = kwargs.pop("output_attentions", False)
#         self.output_attentions = kwargs.pop("output_attentions", False)
#         self.output_attentions = kwargs.pop("output_attentions", False)
#         self.output_attentions = kwargs.pop("output_attentions", False)
#         self.output_attentions = kwargs.pop("output_attentions", False)
#         self.output_attentions = kwargs.pop("output_attentions", False)
#
#         # Additional attributes without default values
#         for key, value in kwargs.items():
#             try:
#                 setattr(self, key, value)
#             except AttributeError as err:
#                 logger.error("Can't set {} with value {} for {}".format(key, value, self))
#                 raise err
# String-to-object mappings

# Maps from my name for models to huggingface shortcut names
CONFIG_MAPPING = OrderedDict(
    [
        ("huggingface_baseline_encoder", "roberta-base",),
    ]
)
from objectives import *

OBJECTIVE_MAPPING = OrderedDict(
    [
        ("t5_mlm", t5_denoise_spans_objective,),
        ("simple_mlm", BERT_MLM_objective,),
    ]
)
