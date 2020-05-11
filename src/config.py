import logging
from collections import OrderedDict
from transformers import AlbertTokenizer

import sys
from datetime import datetime
from absl import flags
# %% FLAGS
from pathlib2 import Path
from transformers import RobertaTokenizer

from constants import READ_ONLY_ROOT, WRITE_ROOT
from my_utils.flag_util import DefaultOrderedDict, get_gpus_with_enough_memory

logger = logging.getLogger(__name__)

#Flags that should be taken from a loaded model's original flags
MODEL_RELEVANT_FLAGS = ["model",
                        "DIR",
                        "d_emb",
                        "d_hidden",
                        "d_ff",
                        "nb_heads",
                        "max_seq_length",
                        "nb_encoder_layers",
                        "nb_feedforward_layers",
                        "relative_attention_num_buckets",
                        "hf_model_handle"]
# TODO maybe make multiple configs? Or maybe keep model hyperparams in some config, and use FLAGS just for folder names etc
FLAGS = flags.FLAGS
flags.DEFINE_integer("d_batch", 2, "Batch size. If DIR is not none, this is also the number of negative samples + 1")
flags.DEFINE_float("DIR_loss_fraction",0.95,"Fraction of the total loss that the distributed regression loss accounts for")
flags.DEFINE_integer("model_save_interval", 300, "Number of seconds after which a model will be checkpointed, even within an epoch")
flags.DEFINE_float("dropout_rate", .1, "Dropout rate")
flags.DEFINE_string("mode", "", "Flag to allow python console command line argument")
flags.DEFINE_bool("mini", False, "Whether to work with mini data for debugging purposes")
flags.DEFINE_integer("beam_width", 3, "Width of the beam during the decoding beam search phase.")
flags.DEFINE_string("data_folder", Path(READ_ONLY_ROOT,"data/Gutenberg").as_posix(), "Folder with train, val and test subfolders containing data")
flags.DEFINE_string("output_folder", Path(WRITE_ROOT,"output","pretraining").as_posix(), "Folder with trained models and tensorboard logs")
flags.DEFINE_string("run_name", datetime.now().strftime("%b_%d_%Hh%Mm%Ss"),
                    "Folder with trained models and tensorboard logs")
flags.DEFINE_integer("relative_attention_num_buckets", 32, "Number of different position embeddings.")
flags.DEFINE_integer("num_serialized_models_to_keep", 1, "Number of serialized trained models to store.")
flags.DEFINE_bool("use_pretrained_weights", False, "Whether to initialize the model with hf pretrained weights.")
flags.DEFINE_string("hf_model_handle","albert-base-v1","Name of the huggingface model handle to use for both tokenizer "
                                                         "and pretrained weights (if those are loaded") #v2 seems to be faulty: https://github.com/huggingface/transformers/pull/1683#issuecomment-556001607
flags.DEFINE_bool("fresh_data",False,"If True, don't use a pickled version of the data input if that existed")
flags.DEFINE_string("saved_pretrained_model_path","",
                    "Path to a checkpoint of a pretrained model. "
                    "If \"pretrained_model\" flag is provided, equals WRITE_ROOT/output/my_model/<pretrained_model>/best.th")
flags.DEFINE_string("cache_dir",Path(READ_ONLY_ROOT,"cache").as_posix(),"Directory to store a cache of 🤗 transformers tokenizer ")
flags.DEFINE_string("description","","Informal description of a run, will be stored in description.txt in the run_name folder")
flags.DEFINE_integer("SG_max_data_size",-1,"If negative, the full data is used for each task. "
                                        "If positive, this is the maximum index up to which samples are considered per epoch "
                                        "during SG finetuning, validating and evaluating")
flags.DEFINE_integer("manual_seed",None,"Running multiple experiments with the same manual seed should give identical performance"
                                        " (barring some unavoidable noise, see https://pytorch.org/docs/stable/notes/randomness.html")
# Trainer flags
flags.DEFINE_integer("patience", 10, "Number of epochs the validation metric can worsen before stopping training.")
flags.DEFINE_integer("num_epochs", 5, "Number of epochs to train for.") # Default low because biggg epochs. Also keeping every epoch for this reason
flags.DEFINE_float("learning_rate", 10e-8, "Learning rate")
flags.DEFINE_integer("nb_subepochs", 100, "Number of points at which to do the logging and validating that the default"
                                      "allenNLP trainer only does at the end of every epoch."
                                      "To deal with large epochs")

#Flags determining denoising objective
flags.DEFINE_string("objective", "simple_mlm",
                    "Name of the denoising objective to use (see OBJECTIVE_MAPPING)")
flags.DEFINE_float("masking_fraction", .15, "Fraction of tokens to be masked during MLM pretraining")


# Flags that determine what the model looks like
flags.DEFINE_string("model", "my_model", "Name of the model to use (see MODEL_MAPPING)")
flags.DEFINE_string("DIR",'',"Which variant of distributed internal regression to employ. Options are: combo, top_down, from_projection, or empty if not using DIR (default)")
flags.DEFINE_integer("d_emb", 128, "Size of token encodings before contextualization")
flags.DEFINE_integer("d_hidden", 2048, "Size of token encodings in hidden layers (contextualized)")
flags.DEFINE_integer("d_ff", 8192, "Number of hidden units in feedforward parts of attention blocks")
flags.DEFINE_integer("d_top_down", 8192, "Number of hidden units in top_down regression")
flags.DEFINE_integer("nb_heads", 16, "Number of attention heads")
flags.DEFINE_integer("max_seq_length", 512, "Maximum number of tokens to consider per batch")
flags.DEFINE_integer("nb_encoder_layers", 24, "Number of layers in the encoder.")
flags.DEFINE_integer("nb_feedforward_layers", 2,
                     "Number of layers in the feedforward subcomponents of the transformer.")
flags.DEFINE_string("activation","gelu","Type of nonlinearity to use.")
flags.DEFINE_string("pos_embeddings","absolute","Type of positional encoding to use.")
flags.DEFINE_float("layernorm_eps",10e-12,"Epsilon to use for Layernorm. Different than default to be in sync with HF Albert")
flags.DEFINE_integer("top_down_distance",2,"For internal prediction: number of layers to feed masked internal activations through before using result to predict masked activation")




# Distributed training stuff
flags.DEFINE_list("device_idxs", get_gpus_with_enough_memory(1000), "List of GPU indices. -1 for CPU. Defaults to the GPUs with at least 8000 MiB memory")
flags.DEFINE_integer("max_GPUs", 3, "Maximum number of GPUs to use at the same time.")
flags.DEFINE_integer("world_size",3,"Number of parallel processes. With current AllenNLP Trainer usage, equals number of GPUs used")
flags.DEFINE_integer("local_rank",None,"Needed for DDP. Automatically assigned by torch distributed launcher, and will be used to pick GPU to run on")

# Jiant cl arguments
flags.DEFINE_string("config_file","", "Location of the file that contains the flow-control-config")
flags.DEFINE_string("pretrained_model","","Name of the run whose best checkpoint will be used as pretrained model."
                                       " Ignored if saved_pretrained_model_path is provided.")

flags.DEFINE_string("pretrained_model_checkpoint","best","Which checkpoint in the training history of a particular pretrained_model to pick"
                                                     "Defaults to 'best'.")
flags.DEFINE_bool("reload_indexing",False,"If True, redo the indexing that happens in jiant")
flags.DEFINE_string("overrides","", "String that indicates which parameters from the config_file to override with what")





FLAGS(sys.argv)
def process_flags():
    assert (not FLAGS.manual_seed == 0), "Set a strictly positive manual seed. Zero is counted as not setting a seed."
    FLAGS.device_idxs = [int(idx) for idx in FLAGS.device_idxs][:FLAGS.max_GPUs]
    assert not (FLAGS.pretrained_model and FLAGS.saved_pretrained_model_path), \
        "You should specify only one of \"saved_pretrained_model_path\" and \"saved_pretrained_model_path\""
    if FLAGS.pretrained_model:
        checkpoint = f'{FLAGS.pretrained_model_checkpoint}.th'
        FLAGS.saved_pretrained_model_path = Path(WRITE_ROOT,"output","pretraining",FLAGS.pretrained_model,checkpoint).as_posix()
    # if FLAGS.config:

from objectives import *


OBJECTIVE_MAPPING = OrderedDict(
    [
        ("t5_mlm", t5_denoise_spans_objective,),
        ("simple_mlm", BERT_MLM_objective,),
    ]
)
TOKENIZER = None
def get_my_tokenizer():
    global TOKENIZER # To not reload tokenizer with different calls
    if TOKENIZER is None:
        TOKENIZER = AlbertTokenizer.from_pretrained(FLAGS.hf_model_handle)
    return TOKENIZER
ACTIVATION_MAPPING = OrderedDict(
    [
        ("gelu", t5_denoise_spans_objective,),
        ("relu", BERT_MLM_objective,),
    ]
)