import logging
from collections import OrderedDict

import sys
from datetime import datetime
from absl import flags
# %% FLAGS
from pathlib2 import Path
from transformers import RobertaTokenizer

from constants import READ_ONLY_ROOT, WRITE_ROOT
from util import get_freer_gpu, DefaultOrderedDict, get_gpus_with_enough_memory

logger = logging.getLogger(__name__)


# TODO maybe make multiple configs? Or maybe keep model hyperparams in some config, and use FLAGS just for folder names etc
FLAGS = flags.FLAGS
flags.DEFINE_integer("d_batch", 8, "Batch size. If use_DIR, this is also the number of negative samples + 1")
flags.DEFINE_string("model", "my_model", "Name of the model to use (see MODEL_MAPPING)")
flags.DEFINE_bool("use_DIR",False,"Whether to use distributed internal regression to create additional losses")
flags.DEFINE_float("DIR_loss_fraction",0.95,"Fraction of the total loss that the distributed regression loss accounts for")
flags.DEFINE_string("tokenizer", "", "Which tokenizer to use. Currently everything defaults to RobertaTokenizer :P")
flags.DEFINE_string("objective", "simple_mlm",
                    "Name of the denoising objective to use (see OBJECTIVE_MAPPING)")
flags.DEFINE_integer("d_emb", 72, "Size of token encodings before contextualization")
flags.DEFINE_integer("d_hidden", 768, "Size of token encodings in hidden layers (contextualized)")
flags.DEFINE_integer("d_ff", 3072, "Number of hidden units in feedforward parts of attention blocks")
flags.DEFINE_integer("model_save_interval", 300, "Number of seconds after which a model will be checkpointed, even within an epoch")
flags.DEFINE_integer("nb_heads", 8, "Number of attention heads")
flags.DEFINE_float("masking_fraction", .15, "Fraction of tokens to be masked during MLM pretraining")
flags.DEFINE_float("dropout_rate", .1, "Dropout rate")
flags.DEFINE_float("learning_rate", 10e-6, "Learning rate")
flags.DEFINE_integer("max_seq_length", 512, "Maximum number of words to consider per batch")
flags.DEFINE_string("data_folder", Path(READ_ONLY_ROOT,"data/Gutenberg").as_posix(), "Folder with train, val and test subfolders containing data")
flags.DEFINE_string("model_folder", Path(WRITE_ROOT,"output").as_posix(), "Folder with trained models and tensorboard logs")
flags.DEFINE_string("run_name", datetime.now().strftime("%b_%d_%Hh%Mm%Ss"),
                    "Folder with trained models and tensorboard logs")
flags.DEFINE_string("mode", "", "Flag to allow python console command line argument")
# Trainer flags
flags.DEFINE_integer("patience", 500, "Number of epochs the validation metric can worsen before stopping training.")
flags.DEFINE_integer("num_epochs", 10000, "Number of epochs to train for.")

flags.DEFINE_bool("mini", False, "Whether to work with mini data/models for debugging purposes")

flags.DEFINE_bool("use_decoder", False, "Whether to use a Transformer decoder on top of the encoder")

flags.DEFINE_integer("nb_encoder_layers", 12, "Number of layers in the encoder.")
flags.DEFINE_integer("nb_decoder_layers", 6, "Number of layers in the decoder.")
flags.DEFINE_integer("nb_feedforward_layers", 2,
                     "Number of layers in the feedforward subcomponents of the transformer.")
flags.DEFINE_integer("relative_attention_num_buckets", 32, "Number of different position embeddings.")
flags.DEFINE_integer("beam_width", 3, "Width of the beam during the decoding beam search phase.")
flags.DEFINE_integer("num_serialized_models_to_keep", 1, "Number of serialized trained models to store.")
flags.DEFINE_bool("use_pretrained_weights", False, "Whether to initialize weights with pretrained weights. "
                                                  "If so, the CONFIG_MAPPING is used to determine weights. "
                                                  "Only works for hf_baseline so far ;)") #TODO maybe expand this to own model
flags.DEFINE_bool("fresh_data",False,"If True, don't use a pickled version of the data input if that existed")


# Distributed training stuff
flags.DEFINE_list("device_idxs", get_gpus_with_enough_memory(3000), "List of GPU indices. -1 for CPU. Defaults to the GPUs with at least 8000 MiB memory")
flags.DEFINE_integer("max_GPUs", 3, "Maximum number of GPUs to use at the same time.")
flags.DEFINE_integer("world_size",3,"Number of parallel processes. With current AllenNLP Trainer usage, equals number of GPUs used")
flags.DEFINE_integer("local_rank",None,"Needed for DDP. Automatically assigned by torch distributed launcher, and will be used to pick GPU to run on")

FLAGS(sys.argv)
FLAGS.device_idxs = FLAGS.device_idxs[:FLAGS.max_GPUs]

#TODO adapt this to per-experiment configs

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
TOKENIZER_MAPPING = DefaultOrderedDict(
    lambda: RobertaTokenizer.from_pretrained(CONFIG_MAPPING['huggingface_baseline_encoder']),
    [
    ]
    )