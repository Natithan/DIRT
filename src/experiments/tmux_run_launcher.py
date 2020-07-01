import time

import libtmux
from numpy.ma import arange
import pickle
import os.path
import subprocess

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from pprint import pprint

import git
import time

from config import FLAGS

RUNS = {}
BASE_SERVER = "arwen"

#
# current_run_name = "HFRoberta_HFpre_nomypre_2"
# current_description = "Re-running the roberta baseline to (hopefully) have the results be stored in my results sheet, " \
#                       "including the total average score on validation data. " \
#                       "That average can then be compared to leaderboard. (this is pure roberta directly training on target SG tasks)"
# RUNS[current_run_name] = [
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f' --max_GPUs=1 '
#         f' --description="{current_description}"'
#         f' --overrides "'
#         f' run_name={current_run_name},'
#         f' input_module=roberta-base'
#         f'"; cd ..'
#     ]

# current_run_name = "HFAlbert_HFpre_nomypre_2"
# current_description = "Re-running the albert baseline (in it's small version that I use) " \
#                       "to (hopefully) have the results be stored in my results sheet, including the total average score." \
#                       "That average can then be compared to roberta baseline"
#
# RUNS[current_run_name] = [
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f' --max_GPUs=1 '
#         f' --description="{current_description}"'
#         f' --overrides "'
#         f' run_name={current_run_name},'
#         f' input_module={FLAGS.hf_model_handle}'
#         f'"; cd ..'
#     ]


# current_run_name = "baseline_HFpre_nomypre_2"
# RUNS[current_run_name] = [
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f" --use_HFpretrained_weights "
#         f'--max_GPUs=1 '
#         f' --'
#         f'--overrides "run_name={current_run_name},'
#         f'input_module=dirt'
#         f'"; cd ..'
#     ]


# current_run_name = "baseline_HFpre_mypre"
# RUNS[current_run_name] = [
#         f"python pretrain.py --max_GPUs=1 --d_batch=2 "
#         f" --use_HFpretrained_weights "
#         f" --run_name={current_run_name}"
#         f" --learning_rate=0.00000001"
#         f" --description='HFpretrained my baseline WITH mypretrain -> check vs my baseline with no mypretrain, form baseline for DIRT alts. "
#         f"Now with smaller learning rate'",
#
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f' --pretrained_model={current_run_name} --max_GPUs=1 '
#         f' --overrides "run_name={current_run_name},input_module=dirt"; cd ..'
#     ]

# current_run_name = "baseline_noHFpre_mypre_lr10e-8"
# RUNS[current_run_name] = [
#         f"python pretrain.py --max_GPUs=1 --d_batch=2 "
#         f" --run_name={current_run_name}"
#         f" --hf_model_handle=albert-base-v2"
#         f" --learning_rate=10e-8"
#         f" --d_hidden=768"
#         f" --d_ff=3072"
#         f" --d_top_down=3072"
#         f" --nb_heads=12"
#         f" --nb_encoder_layers=12"
#         f" --d_batch=3"
#         f" --description='From scratch my Albert with mypretrain -> form baseline for DIRT alts, aiming-for-relative-improvements. "
#         f"With learning rate same as the one that was successful for baseline_HFpre_mypre'",
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f' --pretrained_model={current_run_name} --max_GPUs=1 '
#         f' --overrides "run_name={current_run_name},input_module=dirt"; cd ..'
#     ]
# current_run_name = "HFAlbert_noHFpre_mypre"
# RUNS[current_run_name] = [
#         f"python pretrain.py --max_GPUs=1 --d_batch=2 "
#         f" --run_name={current_run_name}"
#         f"--description='From scratch HF Albert with mypretrain -> to check if my albert is legit, when testing from-scratch'",
#
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f'--pretrained_model={current_run_name} --max_GPUs=1 '
#         f'--overrides "run_name={current_run_name},input_module={FLAGS.hf_model_handle}"; cd ..'
#     ]

# current_run_name = "combo_noHFpre_mypre_2"
# current_description = "Re-running my combo DIRT alt, this time with hopefully no fail on first validation during pretrainging," \
#                       "and a slightly bigger initial learning rate again"
# RUNS[current_run_name] = [
#         f"python pretrain.py --max_GPUs=1 --d_batch=3 --patience=1"
#         f" --run_name={current_run_name}"
#         f' --description="{current_description}"'
#         f" --DIR=combo"
#         f" --d_hidden=768"
#         f" --learning_rate=10e-6"
#         f" --d_ff=3072"
#         f" --d_top_down=3072"
#         f" --nb_heads=12"
#         f" --nb_encoder_layers=12",
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f'--pretrained_model={current_run_name} --max_GPUs=1 '
#         f'--overrides "run_name={current_run_name},input_module=dirt"; cd ..'
#     ]
#
# current_run_name = "vanilla_noHFpre_mypre"
# current_description = "Re-running my from scratch vanilla to form baseline for combo DIRT alt. Also with again slightly " \
#                       "bigger initial learning rate, and small-albert shape."
# RUNS[current_run_name] = [
#         f"python pretrain.py --max_GPUs=1 --d_batch=3 --patience=1"
#         f" --run_name={current_run_name}"
#         f' --description="{current_description}"'
#         f" --d_hidden=768"
#         f" --learning_rate=10e-6"
#         f" --d_ff=3072"
#         f" --d_top_down=3072"
#         f" --nb_heads=12"
#         f" --nb_encoder_layers=12",
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f'--pretrained_model={current_run_name} --max_GPUs=1 '
#         f'--overrides "run_name={current_run_name},input_module=dirt"; cd ..'
#     ]

# current_run_name = "top_down_HFpre_mypre"
# RUNS[current_run_name] = [
#         f"python pretrain.py --max_GPUs=1 --d_batch=2 "
#         f" --run_name={current_run_name}"
#         f" --use_HFpretrained_weights "
#         f"--DIR=top_down"
#         f"--description='With HFpretrain my preffered DIRT alt (aka top_down) with mypretrain -> check if improvement somewhere vs my albert, aiming-for-absolute-high'",
#
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f'--pretrained_model={current_run_name} --max_GPUs=1 '
#         f'--overrides "run_name={current_run_name},input_module=dirt"; cd ..'
#     ]
# current_run_name = "top_down_noHFpre_nomypre"
# RUNS[current_run_name] = [
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f'--max_GPUs=1 '
#         f'--DIR=top_down'
#         f'--overrides "run_name={current_run_name},input_module=dirt"; cd ..'
#     ]
# current_run_name = "baseline_noHFpre_nomypre"
# RUNS[current_run_name] = [
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f'--max_GPUs=1 '
#         f'--overrides "run_name={current_run_name},input_module=dirt"; cd ..'
#     ]

# current_run_name = "combo_HFpre_mypre"
# current_description = "- See if pretrained weights help in doing self-regression    \r\n" \
#                       "- See if doing extra training with DIRT objective on top of pretrained weights improves (any aspect of) GLUE performance"
# current_server = 'arwen'
# RUNS[current_run_name] = [
#         f"ssh {current_server}",
#         f"python pretrain.py --max_GPUs=1 --d_batch=3 --patience=1"
#         f" --run_name={current_run_name}"
#         f' --description="{current_description}"'
#         f' --use_HFpretrained_weights'
#         f" --DIR=combo"
#         f" --d_hidden=768"
#         f" --learning_rate=10e-6"
#         f" --d_ff=3072"
#         f" --d_top_down=3072"
#         f" --nb_heads=12"
#         f" --nb_encoder_layers=12",
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f'--pretrained_model={current_run_name} --max_GPUs=1 '
#         f'--overrides "run_name={current_run_name},input_module=dirt"; cd ..'
#     ]

# current_run_name = "combo_fraction_.5"
# current_description = "Check effect of relative importance of DIRT loss in pretraining task vs default .95"
# current_server = 'frodo'
#
# RUNS[current_run_name] = [
#     f"ssh {current_server}",
#     f"python pretrain.py --max_GPUs=1 --d_batch=3 --patience=1"
#         f" --run_name={current_run_name}"
#         f' --description="{current_description}"'
#         f" --DIR=combo"
#         f" --d_hidden=768"
#         f" --learning_rate=10e-6"
#         f" --d_ff=3072"
#         f" --d_top_down=3072"
#         f" --nb_heads=12"
#         f" --nb_encoder_layers=12",
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f'--pretrained_model={current_run_name} --max_GPUs=1 '
#         f'--overrides "run_name={current_run_name},input_module=dirt"; cd ..'
#     ]

# current_server = 'bilbo'
# current_run_name = "HFRoberta_big_HFpre_nomypre"
# current_description = "Running big roberta variant to see effect on SG score of size"
# RUNS[current_run_name] = [
#         f"ssh {current_server}",
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f' --max_GPUs=1 '
#         f' --description="{current_description}"'
#         f' --overrides "'
#         f' run_name={current_run_name},'
#         f' input_module=roberta-large'
#         f'"; cd ..'
#     ]
# current_server = 'rose'
# current_run_name = "HFRoberta_bigMNLI_HFpre_nomypre"
# current_description = "Running big roberta variant with mnli extra training, to see effect on SG score of size"
# RUNS[current_run_name] = [
#     f"ssh {current_server}",
#     f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f' --max_GPUs=1 '
#         f' --description="{current_description}"'
#         f' --overrides "'
#         f' run_name={current_run_name},'
#         f' input_module=roberta-large-mnli'
#         f'"; cd ..'
#     ]
#
# current_server = 'rose'
# current_run_name = "HFAlbert_big_HFpre_nomypre"
# current_description = "Running big albert variant to see effect on SG score of size"
# RUNS[current_run_name] = [
#     f"ssh {current_server}",
#     f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f' --max_GPUs=1 '
#         f' --description="{current_description}"'
#         f' --overrides "'
#         f' run_name={current_run_name},'
#         f' input_module=albert-xxlarge-v1'
#         f'"; cd ..'
#     ]
#
# current_server = 'frodo'
# current_run_name = "HFAlbert_HFpre_mypre_lr_10emin8"
# current_description = "Noticed that 'baseline_HFpre_mypre_lr_10emin8' scored quite worse than 'HFAlbert_HFpre_nomypre_2'. " \
#                       "This run is the same as 'baseline_HFpre_mypre_lr_10emin8', but with HFAlbert. " \
#                       "It checks if the performance hit is due to my baseline. " \
#                       "If not, it is probs due to the pretraining data/objective. This starts what was interrupted on arwen again on frodo."
# hf_model_handle='albert-xlarge-v1'
# RUNS[current_run_name] = [
#         f"ssh {current_server}",
#
# f"python pretrain.py --max_GPUs=1 --d_batch=2 --patience=1"
# f" --run_name={current_run_name}"
# f' --description="{current_description}"'
# f' --model=hf_baseline'
# f' --use_HFpretrained_weights'
# f' --hf_model_handle={hf_model_handle}'
# f' --learning_rate=0.0000001',

#
#     f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#     f' --max_GPUs=1 '
#     f' --description="{current_description}"'
#     f' --saved_pretrained_model_path={Path(READ_ONLY_ROOT,"output","pretraining",current_run_name,"best.th").as_posix()}'
#     f' --overrides "'
#     f' run_name={current_run_name},'
#     f' input_module=dirt'
#     f'"; cd ..'
# ]


# current_server = 'bilbo'
# current_run_name = "vanilla_noHFpre_mypre_3"
# current_description = "Re-running my from scratch vanilla to form baseline for combo DIRT alt. " \
#                       "THIS TIME with saving 5 USEFUL checkpoints, and training for max 5 epochs."
# RUNS[current_run_name] = [
#     f"ssh {current_server}",
#
#     f"python pretrain.py --max_GPUs=1 --d_batch=3 --patience=1"
#         f" --run_name={current_run_name}"
#         f' --description="{current_description}"'
#         f" --d_hidden=768"
#         f" --learning_rate=10e-6"
#         f" --d_ff=3072"
#         f" --d_top_down=3072"
#         f" --nb_heads=12"
#         f" --nb_encoder_layers=12"
#         f" --num_epochs=5"
#         f" --patience=5"
#         f" --num_serialized_models_to_keep=5",
#
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f'--pretrained_model={current_run_name} --max_GPUs=1 '
#         f'--overrides "run_name={current_run_name},input_module=dirt"; cd ..'
#     ]

# current_server = 'bilbo'
# current_run_name = "combo_noHFpre_mypre_4_one_epoch"
# pretrained_model= "combo_noHFpre_mypre_4"
# pretrained_model_checkpoint = f'model_state_epoch_0'
# current_description = "SG-evaluating the combo_noHFpre_mypre_4 run after 1 epoch of pretraining"
# current_size = 'base'
# RUNS[current_run_name] = [
#         f"ssh {current_server}",
#
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f'--pretrained_model={pretrained_model} --max_GPUs=1 '
#         f' --pretrained_model_checkpoint={pretrained_model_checkpoint}'
#         f' --overrides "run_name={current_run_name},input_module=dirt"; cd ..'
#     ]
# current_server = 'bilbo'
# current_run_name = "combo_HFpre_mypre"
# current_description = "See if pretrained weights help in doing self-regression    \r\n" \
#                       "See if doing extra training with DIRT objective on top of pretrained weights improves (any aspect of) GLUE performance"
# RUNS[current_run_name] = [
#         f"ssh {current_server}",
#
#         f"python pretrain.py --max_GPUs=1 --d_batch=3 --patience=1"
#         f" --run_name={current_run_name}"
#         f' --description="{current_description}"'
#         f' --use_HFpretrained_weights'
#         f" --DIR=combo"
#         f" --flagfile=configs/base.txt",
#
#
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f' --pretrained_model={current_run_name} --max_GPUs=1 '
#         f' --overrides "run_name={current_run_name},input_module=dirt"; cd ..'
#     ]

# current_run_name = "combo_fraction_.5"
# current_description = "Check effect of relative importance of DIRT loss in pretraining task vs default .95"
# current_server = 'bilbo'
#
# RUNS[current_run_name] = [
#     f"ssh {current_server}",
#
#     f"python pretrain.py --max_GPUs=1 --d_batch=3 --patience=1"
#     f" --run_name={current_run_name}"
#     f' --description="{current_description}"'
#     f" --DIR=combo"
#     f" --flagfile=configs/base.txt"
#     f" --DIR_loss_fraction=0.5",
#
#     f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#     f' --pretrained_model={current_run_name} --max_GPUs=1 '
#     f' --overrides "run_name={current_run_name},input_module=dirt"; cd ..'
#     ]

# current_run_name = "combo_fraction_.1"
# current_description = "Check effect of relative importance of DIRT loss in pretraining task vs default .95"
# current_server = 'bilbo'
#
# RUNS[current_run_name] = [
#     f"ssh {current_server}",
#
#     f"python pretrain.py --max_GPUs=1 --d_batch=3 --patience=1"
#     f" --run_name={current_run_name}"
#     f' --description="{current_description}"'
#     f" --DIR=combo"
#     f" --flagfile=configs/base.txt"
#     f" --DIR_loss_fraction=0.5",
#
#     f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#     f' --pretrained_model={current_run_name} --max_GPUs=1 '
#     f' --overrides "run_name={current_run_name},input_module=dirt"; cd ..'
#     ]

#
# current_run_name = "HFRoberta_bigMNLI_HFpre_nomypre"
# current_description = "Running big roberta variant with mnli extra training, to see effect on SG score of size"
# current_server = 'bilbo'
#
# RUNS[current_run_name] = [
#     f"ssh {current_server}",
#
#     f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#     f'--max_GPUs=1 '
#     f' --overrides "run_name={current_run_name},input_module=roberta-large-mnli"; cd ..'
#     ]

# current_run_name = "HFRoberta_big_HFpre_nomypre_2"
# current_description = "Running big roberta variant to see effect on SG score of size. This time correct (instead of actually mnli version also)"
# current_server = 'frodo'
#
# RUNS[current_run_name] = [
#     f"ssh {current_server}",
#
#     f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#     f'--max_GPUs=1 '
#     f' --overrides "run_name={current_run_name},input_module=roberta-large"; cd ..'
#     ]
#
# current_run_name = "HFAlbert_big_HFpre_nomypre"
# current_description = "Running big albert variant to see effect on SG score of size"
# current_server = 'frodo'
#
# RUNS[current_run_name] = [
#     f"ssh {current_server}",
#
#     f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#     f'--max_GPUs=1 '
#     f' --overrides "run_name={current_run_name},input_module=albert-xxlarge-v2"; cd ..'
#     ]

# current_server = 'arwen'
# current_run_name = "HFAlbert_xl_HFpre_mypre_lr_10emin8_2"
# current_description = "After fixing my baseline to be equal to HF in dropouts also, this checks whether my " \
#                       "baseline can indeed can match HFAlbert with both HFpretrain and my_pretrain." \
#                       "This time pretraining with 5 epochs patience, for max 5 epochs."
# hf_model_handle='albert-xlarge-v1'
# RUNS[current_run_name] = [
#         f"ssh {current_server}",
#
#         f"python pretrain.py --max_GPUs=1 --d_batch=2 --patience=1"
#         f" --run_name={current_run_name}"
#         f' --description="{current_description}"'
#         f' --model=hf_baseline'
#         f" --flagfile=configs/xlarge.txt"
#         f' --use_HFpretrained_weights'
#         f' --hf_model_handle={hf_model_handle}'
#         f' --learning_rate=0.0000001',
#
#
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f' --max_GPUs=1 '
#         f' --description="{current_description}"'
#         f' --pretrained_model={current_run_name}'
#         f' --overrides "'
#         f' run_name={current_run_name},'
#         f' input_module=dirt'
#         f'"; cd ..'
#     ]
#
# current_server = 'bilbo'
# current_run_name = "baseline_xl_HFpre_mypre_lr_10emin8_2"
# current_description = "After fixing my baseline to be equal to HF in dropouts also, this checks whether my baseline can" \
#                       " indeed match HFAlbert with both HFpretrain and my_pretrain. " \
#                       "This time pretraining with 5 epochs patience, for max 5 epochs."
# hf_model_handle='albert-xlarge-v1'
# RUNS[current_run_name] = [
#         f"ssh {current_server}",
#
#         f"python pretrain.py --max_GPUs=1 --d_batch=2 "
#         f" --patience=5"
#         f" --run_name={current_run_name}"
#         f' --description="{current_description}"'
#         f" --flagfile=configs/xlarge.txt"
#         f' --use_HFpretrained_weights'
#         f' --hf_model_handle={hf_model_handle}'
#         f' --learning_rate=0.0000001',
#
#
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f' --max_GPUs=1 '
#         f' --description="{current_description}"'
#         f' --pretrained_model={current_run_name}'
#         f' --overrides "'
#         f' run_name={current_run_name},'
#         f' input_module=dirt'
#         f'"; cd ..'
#     ]
#
# current_server = 'frodo'
# current_run_name = "baseline_base_HFpre_mypre_lr_10emin8_2"
# current_description = "Forming baseline with HFPre, with updated dropout."
# hf_model_handle='albert-base-v1'
# RUNS[current_run_name] = [
#         f"ssh {current_server}",
#
#         f"python pretrain.py --max_GPUs=1 --d_batch=2 "
#         f" --patience=5"
#         f" --run_name={current_run_name}"
#         f' --description="{current_description}"'
#         f" --flagfile=configs/base.txt"
#         f' --use_HFpretrained_weights'
#         f' --hf_model_handle={hf_model_handle}'
#         f' --learning_rate=0.0000001'
#         f' --device_idxs=3',
#
#
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f' --max_GPUs=1 '
#         f' --device_idxs=3'
#         f' --description="{current_description}"'
#         f' --pretrained_model={current_run_name}'
#         f' --overrides "'
#         f' run_name={current_run_name},'
#         f' input_module=dirt'
#         f'"; cd ..'
#     ]

# current_server = 'frodo'
# current_run_name = "HFAlbert_xl_HFpre_mypre_lr_10emin8_2_mid_epoch_check"
# current_description = "This checks SG performance with the already existing checkpoint, if this is already decent and matches baseline, I'll be satisfied "
# hf_model_handle='albert-xlarge-v1'
# RUNS[current_run_name] = [
#         f"ssh {current_server}",
#
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f' --max_GPUs=1 '
#         f' --description="{current_description}"'
#         f' --saved_pretrained_model_path={Path("/cw/working-arwen/nathan/phd","output","pretraining","HFAlbert_xl_HFpre_mypre_lr_10emin8_2","model_state_epoch_0.2020-05-05-10-10-30.th").as_posix()}'
#         f' --overrides "'
#         f' run_name={current_run_name},'
#         f' input_module=dirt'
#         f'"; cd ..'
#     ]
#
# current_server = 'frodo'
# current_run_name = "baseline_xl_HFpre_mypre_lr_10emin8_2_mid_epoch_check"
# current_description = "This checks SG performance with the already existing checkpoint, if this is already decent and matches HFalbert, I'll be satisfied "
# hf_model_handle='albert-xlarge-v1'
# RUNS[current_run_name] = [
#         f"ssh {current_server}",
#
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f' --max_GPUs=1 '
#         f' --description="{current_description}"'
#         f' --saved_pretrained_model_path={Path("/cw/working-bilbo/nathan/phd","output","pretraining","baseline_xl_HFpre_mypre_lr_10emin8_2","model_state_epoch_0.2020-05-05-10-12-30.th").as_posix()}'
#         f' --overrides "'
#         f' run_name={current_run_name},'
#         f' input_module=dirt'
#         f'"; cd ..'
#     ]
# current_server = 'frodo'
# current_run_name = "vanilla_noHFpre_mypre_4"
# current_description = "A baseline run to compare with combo_noHFpre_mypre_5. With updated code: dropouts now completely as in HFAlbert, and updated SG training data: keeping a separate held-out set from the train data"
# RUNS[current_run_name] = [
#         f"ssh {current_server}",
#
#         f"python pretrain.py --max_GPUs=1 --d_batch=3"
#             f" --run_name={current_run_name}"
#             f' --description="{current_description}"'
#             f" --flagfile=configs/base.txt"
#             f" --learning_rate=10e-6"
#             f" --num_epochs=5"
#             f" --patience=6"
#             f" --num_serialized_models_to_keep=1"
#         f" --device_idxs=2",
#
#             f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#             f' --pretrained_model={current_run_name} --max_GPUs=1  --device_idxs=2'
#             f' --overrides "run_name={current_run_name}"; cd ..'
#         ]
# current_server = 'bilbo'
# current_run_name = "combo_noHFpre_mypre_5"
# current_description = "A DIRT run to compare with vanilla_noHFpre_mypre_4. With updated code: dropouts now completely as in HFAlbert, and updated SG training data: keeping a separate held-out set from the train data"
# RUNS[current_run_name] = {'commands':[
#         f"ssh {current_server}",
#
#         # f"python pretrain.py --max_GPUs=1 --d_batch=3 "
#         # f" --DIR=combo"
#         #     f" --run_name={current_run_name}"
#         #     f' --description="{current_description}"'
#         #     f" --flagfile=configs/base.txt"
#         #     f" --learning_rate=10e-6"
#         #     f" --num_epochs=5"
#         #     f" --patience=6"
#         #     f" --num_serialized_models_to_keep=1"
#         # f" --device_idxs=3",
#
#             f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#             f' --pretrained_model={current_run_name} --max_GPUs=1 '
#             f' --overrides "run_name={current_run_name}"; cd ..'
#         ],
#     'description': current_description,
#     'server': current_server}


# current_server = 'frodo'
# current_run_name = "vanilla_HFpre_mypre_2"
# current_description = "A DIRT run with most recent code to compare with HFPretrain weights between this and combo"
# RUNS[current_run_name] = [
#         f"ssh {current_server}",
#
#         f"python pretrain.py --max_GPUs=1 --d_batch=3  --device_idxs=2"
#             f" --run_name={current_run_name}"
#             f' --description="{current_description}"'
#             f" --flagfile=configs/base.txt"
#             f" --learning_rate=10e-6"
#             f" --num_epochs=5"
#             f" --patience=6"
#             f" --num_serialized_models_to_keep=1"
#         f" --use_HFpretrained_weights",
#
#             f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#             f' --pretrained_model={current_run_name} --max_GPUs=1  --device_idxs=2 '
#             f' --overrides "run_name={current_run_name}"; cd ..'
#         ]
# current_server = 'bilbo'
# current_run_name = "combo_HFpre_mypre_2"
# current_description = "A DIRT run with most recent code to compare with HFPretrain weights between this and vanilla"
# RUNS[current_run_name] = [
#         f"ssh {current_server}",
#
#         f"python pretrain.py --max_GPUs=1 --d_batch=3"
#         f" --DIR=combo"
#             f" --run_name={current_run_name}"
#             f' --description="{current_description}"'
#             f" --flagfile=configs/base.txt"
#             f" --learning_rate=10e-6"
#             f" --num_epochs=5"
#             f" --patience=6"
#             f" --num_serialized_models_to_keep=1"
#         f" --use_HFpretrained_weights",
#
#             f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#             f'--pretrained_model={current_run_name} --max_GPUs=1 '
#             f'--overrides "run_name={current_run_name}"; cd ..'
#         ]
# current_server = 'bilbo'
# current_run_name = "combo_HFpre_mypre_2"
# current_description = "A DIRT run with most recent code to compare with HFPretrain weights between this and vanilla"
# RUNS[current_run_name] = {'commands': [
#     f"ssh {current_server}",
#
#     # f"python pretrain.py --max_GPUs=1 --d_batch=3"
#     # f" --DIR=combo"
#     # f" --run_name={current_run_name}"
#     # f' --description="{current_description}"'
#     # f" --flagfile=configs/base.txt"
#     # f" --learning_rate=10e-6"
#     # f" --num_epochs=5"
#     # f" --patience=6"
#     # f" --num_serialized_models_to_keep=1"
#     # f" --use_HFpretrained_weights",
#
#     f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#     f'--pretrained_model={current_run_name} --max_GPUs=1 '
#     f'--overrides "run_name={current_run_name}"; cd ..'
# ],
#     'description': current_description,
#     'server': current_server}

# current_server = 'frodo'
# current_run_name = "vanilla_noHFpre_nomypre"
# current_description = "Validate my_pretraining: check that it improves SG performance when compared to from scratch"
# RUNS[current_run_name] = {'commands': [
#     f"ssh {current_server}",
#
#
#     f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#     f' --max_GPUs=1 '
#     f' --overrides "run_name={current_run_name}"; cd ..'
# ],
#     'description': current_description,
#     'server': current_server}
#


# current_server = 'frodo'
# current_run_name = "vanilla_HFpre_nomypre"
# current_description = "Validate my_pretrain: check for baseline that additional my_pretrain at least doesn't deteriorate performance." \
#                       " Serve as basis to compare with vanilla__HFpre_mypre"
# RUNS[current_run_name] = {'commands': [
#     f"ssh {current_server}",
#
#     f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#     f' --max_GPUs=1 '
#     f' --use_pretrained_weights'
#     f' --flagfile=../configs/base.txt'
#     f' --overrides "run_name={current_run_name}"; cd ..'
# ],
#     'description': current_description,
#     'server': current_server}

# current_server = 'frodo'
# current_run_name = "combo_noHFpre_mypre_5_r2"
# current_description = "A second DIRT run, just to have an average"
# RUNS[current_run_name] = {'commands': [
#         f"ssh {current_server}",
#
#         f"python pretrain.py --max_GPUs=1 --d_batch=3 "
#         f" --DIR=combo"
#             f" --run_name={current_run_name}"
#             f' --description="{current_description}"'
#             f" --flagfile=configs/base.txt"
#             f" --learning_rate=10e-6"
#             f" --num_epochs=5"
#             f" --patience=6"
#             f" --num_serialized_models_to_keep=1",
#
#             f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#             f' --pretrained_model={current_run_name} --max_GPUs=1 '
#             f' --overrides "run_name={current_run_name}"; cd ..'
#         ],
#     'description': current_description,
#     'server': current_server}
#
# current_server = 'bilbo'
# current_run_name = "vanilla_noHFpre_mypre_4_r2"
# current_description = "A second run of a baseline run to compare with combo_noHFpre_mypre_5, to have an average."
# RUNS[current_run_name] = {'commands': [
#         f"ssh {current_server}",
#
#         f"python pretrain.py --max_GPUs=1 --d_batch=3"
#             f" --run_name={current_run_name}"
#             f' --description="{current_description}"'
#             f" --flagfile=configs/base.txt"
#             f" --learning_rate=10e-6"
#             f" --num_epochs=5"
#             f" --patience=6"
#             f" --num_serialized_models_to_keep=1"
#         f" --device_idxs=2",
#
#             f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#             f' --pretrained_model={current_run_name} --max_GPUs=1  --device_idxs=2'
#             f' --overrides "run_name={current_run_name}"; cd ..'
#         ],
#     'description': current_description,
#     'server': current_server}

# current_server = 'rose'
# current_run_name = "alternating_prediction_noHFpre_mypre"
# current_description = "Test whether passing on predicted values (alternatingly with improving those predictions) " \
#                       "improves performance by virtue of acting as a regularization. To compare with combo_noHFpre_mypre_5"
# RUNS[current_run_name] = {'commands': [
#         f"ssh {current_server}",
#
#         f"python pretrain.py --max_GPUs=1 --d_batch=3 "
#         f" --DIR=combo"
#             f" --run_name={current_run_name}"
#             f' --description="{current_description}"'
#             f" --flagfile=configs/base.txt"
#             f" --learning_rate=10e-6"
#             f" --num_epochs=5"
#             f" --patience=6"
#             f" --num_serialized_models_to_keep=1"
#         f" --alternate_internal_prediction",
#
#             f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#             f' --pretrained_model={current_run_name} --max_GPUs=1 '
#             f' --overrides "run_name={current_run_name}"; cd ..'
#         ],
#     'description': current_description,
#     'server': current_server}

# current_server = 'frodo'
# current_run_name = "alternating_prediction_noHFpre_mypre_r2"
# current_description = "Another run for: Test whether passing on predicted values (alternatingly with improving those predictions) " \
#                       "improves performance by virtue of acting as a regularization. To compare with combo_noHFpre_mypre_5"
# RUNS[current_run_name] = {'commands': [
#         f"ssh {current_server}",
#
#         f"python pretrain.py --max_GPUs=1 --d_batch=3 "
#         f" --DIR=combo"
#             f" --run_name={current_run_name}"
#             f' --description="{current_description}"'
#             f" --flagfile=configs/base.txt"
#             f" --learning_rate=10e-6"
#             f" --num_epochs=5"
#             f" --patience=6"
#             f" --num_serialized_models_to_keep=1"
#         f" --alternate_internal_prediction"
#         f" --old_pretrain_data",
#
#             f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#             f' --pretrained_model={current_run_name} --max_GPUs=1 '
#             f' --overrides "run_name={current_run_name}"; cd ..'
#         ],
#     'description': current_description,
#     'server': current_server}

# current_server = 'arwen'
# current_run_name = "alternating_prediction_noHFpre_mypre_new_data"
# current_description = "Run to test how long one epoch on wiki+BC+GB would take"
# RUNS[current_run_name] = {'commands': [
#         f"ssh {current_server}",
#
#         f"python pretrain.py --max_GPUs=1 --d_batch=3 "
#         f" --DIR=combo"
#             f" --run_name={current_run_name}"
#             f' --description="{current_description}"'
#             f" --flagfile=configs/base.txt"
#             f" --learning_rate=10e-6"
#             f" --num_epochs=5"
#             f" --patience=6"
#             f" --num_serialized_models_to_keep=1"
#         f" --alternate_internal_prediction",
#
#             # f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#             # f' --pretrained_model={current_run_name} --max_GPUs=1 '
#             # f' --overrides "run_name={current_run_name}"; cd ..'
#         ],
#     'description': current_description,
#     'server': current_server}


# for current_server, current_lambda in zip(
#     # ['frodo','frodo','frodo','arwen','rose','rose'],
#     #     [0,.2,.4,.6,.8,1]
#
#         [ 'frodo','frodo', 'rose', 'rose'],
#         [.2, .6, .8, 1]
# ):
#     current_run_name = f"lambda_{current_lambda}_HFpretrain"
#     current_description = "Part of set of runs that test whether adding the DIR objective improves performance at _a_ fraction"
#     RUNS[current_run_name] = {'commands': [
#             f"ssh {current_server}",
#
#             f"conda activate p1;python pretrain.py --max_GPUs=1 --d_batch=5 --max_seq_length=256 "
#             f" --DIR=combo"
#             f" --run_name={current_run_name}"
#             f' --description="{current_description}"'
#             f" --flagfile=configs/base.txt"
#             f" --use_HFpretrained_weights"
#             f" --learning_rate=10e-6"
#             f" --num_epochs=5"
#             f" --patience=6"
#             f" --num_serialized_models_to_keep=1"
#             f" --alternate_internal_prediction"
#             f" --old_pretrain_data"
#             f" --DIR_loss_fraction={current_lambda}",
#
#             f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#             f' --pretrained_model={current_run_name} --max_GPUs=1 '
#             f' --overrides "run_name={current_run_name}"; cd ..'
#             ],
#         'description': current_description,
#         'server': current_server}

# selected_epoch = 2
# for current_server, current_lambda in zip(
#     # ['frodo','frodo','frodo','arwen','rose','rose'],
#     #     [0,.2,.4,.6,.8,1]
#
#         [ 'frodo','frodo', 'arwen','rose', 'rose'],
#         [0, .2, .4, .8, 1]
# ):
#     current_run_name = f"lambda_{current_lambda}_HFpretrain"
#     current_pretrained_model_path = f"/cw/working-{current_server}/nathan/phd/output/pretraining/" \
#                                     f"{current_run_name}/model_state_epoch_{selected_epoch}.th"
#     current_run_name = current_run_name + f'_epoch_{selected_epoch}'
#     current_description = "Early evaluation on SG of runs for different lambdas, to get a result before progress meet"
#     RUNS[current_run_name] = {'commands': [
#             f"ssh {current_server}",
#
#             # f"conda activate p1;python pretrain.py --max_GPUs=1 --d_batch=5 --max_seq_length=256 "
#             # f" --DIR=combo"
#             # f" --run_name={current_run_name}"
#             # f' --description="{current_description}"'
#             # f" --flagfile=configs/base.txt"
#             # f" --use_HFpretrained_weights"
#             # f" --learning_rate=10e-6"
#             # f" --num_epochs=5"
#             # f" --patience=6"
#             # f" --num_serialized_models_to_keep=1"
#             # f" --alternate_internal_prediction"
#             # f" --old_pretrain_data"
#             # f" --DIR_loss_fraction={current_lambda}",
#
#             f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#             f' --saved_pretrained_model_path={current_pretrained_model_path}'
#             f' --max_GPUs=1 '
#             f' --overrides "run_name={current_run_name}"; cd ..'
#             ],
#         'description': current_description,
#         'server': current_server}

# current_server = 'frodo'
# current_run_name = "vanilla_Bigdata_0.1_HFpre_mypre"
# current_description = "Getting a pretrained baseline on a fraction of the shouldbe-better-domain-data that is small enough to train an epoch in 3 days"
# RUNS[current_run_name] = {'commands': [
#         f"ssh {current_server}",
#
#         f"conda activate p1; python pretrain.py --max_GPUs=1 --d_batch=8 "
#             f" --run_name={current_run_name}"
#             f' --description="{current_description}"'
#             f" --flagfile=configs/base.txt"
#             f" --learning_rate=10e-6"
#             f" --num_epochs=1"
#             f" --patience=6"
#             f" --num_serialized_models_to_keep=1"
#         f" --max_seq_length=256"
#         f" --use_HFpretrained_weights",
#
#             f'  cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#             f' --pretrained_model={current_run_name} --max_GPUs=1 '
#             f' --overrides "run_name={current_run_name}"; cd ..'
#         ],
#     'description': current_description,
#     'server': current_server}

# current_server = 'arwen'
# current_run_name = "twoStep_SG_Bigdata_0.1_HFpretrain_mypretrain"
# current_description = "Testing whether this setup can improve results: first training an internal predictor separately" \
#                       " while freezing the main weights, then freezing the internal predictor, and using it to replace" \
#                       " internal states DURING SG FINETUNING"
# RUNS[current_run_name] = {'commands': [
#         f"ssh {current_server}",
#
#         f"python pretrain.py --max_GPUs=1 --d_batch=8 "
#             f" --run_name={current_run_name}"
#             f' --description="{current_description}"'
#             f" --flagfile=configs/base.txt"
#             f" --learning_rate=10e-6"
#             f" --num_epochs=1"
#             f" --patience=6"
#             f" --num_serialized_models_to_keep=1"
#         f" --max_seq_length=256"
#         f" --freeze_main_model"
#         f" --DIR=combo"
#         f" --use_HFpretrained_weights",
#
#
#             f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#             f' --pretrained_model={current_run_name} --max_GPUs=1 '
#             f'  --replace_self_predictions=always'
#             f' --overrides "run_name={current_run_name}'
#         f'"; cd ..'
#         ],
#     'description': current_description,
#     'server': current_server}

# current_server = 'frodo'
# current_run_name = "vanilla_Bigdata_0.1_HFpre_nomypre_noDrop"
# current_description = "An experiment focused only on finetuning stage: comparing whether using a trained self-predictor" \
#                       "as regularization improves over vanilla when dropout is disabled"
# RUNS[current_run_name] = {'commands': [
#         f"ssh {current_server}",
#
#             f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#             f' --max_GPUs=1 '
#             f' --dropout_rate=0 '
#             f' --overrides "run_name={current_run_name},'
#             f'input_module=albert-base-v1'
#         f'"; cd ..'
#         ],
#     'description': current_description,
#     'server': current_server}
#
# current_server = 'frodo'
# current_run_name = "self-predicting_Bigdata_0.1_HFpre_nomypre_noDrop"
# current_description = "An experiment focused only on finetuning stage: comparing whether using a trained self-predictor" \
#                       "as regularization improves over vanilla when dropout is disabled"
# pretrained_model_path = "/cw/working-arwen/nathan/phd/output/pretraining/twoStep_SG_Bigdata_0.1_HFpretrain_mypretrain/best.th"
# RUNS[current_run_name] = {'commands': [
#         f"ssh {current_server}",
#
#         # f"python pretrain.py --max_GPUs=1 --d_batch=8 "
#         #     f" --run_name={current_run_name}"
#         #     f' --description="{current_description}"'
#         #     f" --flagfile=configs/base.txt"
#         #     f" --learning_rate=10e-6"
#         #     f" --num_epochs=1"
#         #     f" --patience=6"
#         #     f" --num_serialized_models_to_keep=1"
#         # f" --max_seq_length=256"
#         # f" --freeze_main_model"
#         # f" --DIR=combo"
#         # f" --use_HFpretrained_weights",
#
#
#             f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#             f' --max_GPUs=1'
#             f' --dropout_rate=0 '
#             f' --saved_pretrained_model_path={pretrained_model_path}'
#             f' --overrides "run_name={current_run_name}'
#         f'"; cd ..'
#         ],
#     'description': current_description,
#     'server': current_server}

# for current_server, current_lambda in zip(
#
#         ['frodo', 'frodo', 'bilbo', 'bilbo'],
#         [0, .3, .6, .9]
# ):
#     current_run_name = f"lambda_{current_lambda}_HFpretrain_WBG"
#     current_description = "Updated to work with wiki+bc+gb data: pretraining with different fractions lambda of" \
#                           " DIR loss objective, seeing if improvement at any lambda." \
#                           "No DAR, only DAO "
#     RUNS[current_run_name] = {'commands': [
#         f"ssh {current_server}",
#
#         f"conda activate p1;python pretrain.py --run_name={current_run_name} --description=\"{current_description}\" "
#         f" --max_GPUs=1 --learning_rate=10e-6 --num_epochs=1 --patience=6 --num_serialized_models_to_keep=1 --flagfile=configs/base.txt"
#         f" --d_batch=5 --max_seq_length=256 "
#         f" --DIR=combo"
#         f" --replace_self_predictions=''"
#         f" --use_HFpretrained_weights"
#         f" --DIR_loss_fraction={current_lambda}",
#
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f' --pretrained_model={current_run_name} --max_GPUs=1 '
#         f' --overrides "run_name={current_run_name}"; cd ..'
#     ],
#         'description': current_description,
#         'server': current_server}

# current_server = 'arwen'
# current_run_name = "twoStep_SG_Bigdata_0.1_HFpretrain_mypretrain_v2"
# current_description = "Testing whether this setup can improve results: first training an internal predictor separately" \
#                       " while freezing the main weights, then freezing the internal predictor, and using it to replace" \
#                       " internal states DURING SG FINETUNING." \
#                       "This time with no handicap during inference, and corrected freezing of main weights (and of " \
#                       "self-predictor if replace_self_predictions=alternate, but that's not relevant for this experiment)"
# RUNS[current_run_name] = {'commands': [
#         f"ssh {current_server}",
#
#         f"python pretrain.py --max_GPUs=1 --d_batch=20 "
#             f" --run_name={current_run_name}"
#             f' --description="{current_description}"'
#             f" --flagfile=configs/base.txt"
#             f" --learning_rate=10e-6"
#             f" --num_epochs=1"
#             f" --patience=6"
#             f" --num_serialized_models_to_keep=1"
#         f" --max_seq_length=256"
#         f" --freeze_main_model"
#         f" --DIR=combo"
#         f" --use_HFpretrained_weights",
#
#
#             f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#             f' --pretrained_model={current_run_name} --max_GPUs=1 '
#             f'  --replace_self_predictions=always'
#             f' --overrides "run_name={current_run_name}'
#         f'"; cd ..'
#         ],
#     'description': current_description,
#     'server': current_server}
# for current_server, current_lambda in zip(
#
#         ['bilbo', 'bilbo'],
#         [0, .3]
# ):
#     current_run_name = f"lambda_{current_lambda}_HFpretrain_WBG_r2"
#     current_description = "A second run (with bigger batch size) of: Updated to work with wiki+bc+gb data: pretraining with different fractions lambda of" \
#                           " DIR loss objective, seeing if improvement at any lambda." \
#                           "No DAR, only DAO "
#     RUNS[current_run_name] = {'commands': [
#         f"ssh {current_server}",
#
#         f"conda activate p1;python pretrain.py --run_name={current_run_name} --description=\"{current_description}\" "
#         f" --max_GPUs=1 --learning_rate=10e-6 --num_epochs=1 --patience=6 --num_serialized_models_to_keep=1 --flagfile=configs/base.txt"
#         f" --d_batch=8 --max_seq_length=256 "
#         f" --DIR=combo"
#         f" --replace_self_predictions=''"
#         f" --use_HFpretrained_weights"
#         f" --DIR_loss_fraction={current_lambda}",
#
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f' --pretrained_model={current_run_name} --max_GPUs=1 '
#         f' --overrides "run_name={current_run_name}"; cd ..'
#     ],
#         'description': current_description,
#         'server': current_server}

# current_server = 'rose'
# current_run_name = "self-predicting_Bigdata_0.1_HFpre_nomypre_noDrop_v2"
# current_description = "With properly-frozen-pretrained-self-predictor: An experiment focused only on finetuning stage: " \
#                       "comparing whether using a trained self-predictor as regularization improves over vanilla when dropout is disabled"
# pretrained_model_path = "/cw/working-arwen/nathan/phd/output/pretraining/twoStep_SG_Bigdata_0.1_HFpretrain_mypretrain_v2/best.th"
# RUNS[current_run_name] = {'commands': [
#         f"ssh {current_server}",
#
#         # f"python pretrain.py --max_GPUs=1 --d_batch=8 "
#         #     f" --run_name={current_run_name}"
#         #     f' --description="{current_description}"'
#         #     f" --flagfile=configs/base.txt"
#         #     f" --learning_rate=10e-6"
#         #     f" --num_epochs=1"
#         #     f" --patience=6"
#         #     f" --num_serialized_models_to_keep=1"
#         # f" --max_seq_length=256"
#         # f" --freeze_main_model"
#         # f" --DIR=combo"
#         # f" --use_HFpretrained_weights",
#
#
#             f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#             f' --max_GPUs=1'
#             f' --dropout_rate=0 '
#             f' --saved_pretrained_model_path={pretrained_model_path}'
#             f' --overrides "run_name={current_run_name}'
#         f'"; cd ..'
#         ],
#     'description': current_description,
#     'server': current_server}

# for current_server, current_lambda in zip(
#
#         ['bilbo', 'bilbo'],
#         [0, .3]
# ):
# current_server = 'rose'
# current_lambda = 0.3
# current_run_name = f"alternate_lambda_{current_lambda}_HFpre_WBG"
# current_description = "Rerun alternating DIRT with WBG data, to compare with before-WBG, and with both vanilla_WBG and combo_WBG "
# RUNS[current_run_name] = {'commands': [
#     f"ssh {current_server}",
#
#     f"conda activate p1;python pretrain.py --run_name={current_run_name} --description=\"{current_description}\" "
#     f" --max_GPUs=1 --learning_rate=10e-6 --num_epochs=1 --patience=6 --num_serialized_models_to_keep=1 --flagfile=configs/base.txt"
#     f" --d_batch=8 --max_seq_length=256 "
#     f" --DIR=combo"
#     f" --replace_self_predictions='alternate'"
#     f" --use_HFpretrained_weights"
#     f" --DIR_loss_fraction={current_lambda}",
#
#     f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#     f' --pretrained_model={current_run_name} --max_GPUs=1 '
#     f' --overrides "run_name={current_run_name}"; cd ..'
# ],
#     'description': current_description,
#     'server': current_server}

# current_server = 'arwen'
# current_lambda = 0.3
# current_run_name = f"combo_small_pred_lambda_{current_lambda}_HFpre_WBG"
# current_description = "Testing whether a smaller predictor (=better for mem) can improve over vanilla / " \
#                       "over bigger self-predictor"
# RUNS[current_run_name] = {'commands': [
#     f"ssh {current_server}",
#
#     f"conda activate p1;python pretrain.py --run_name={current_run_name} --description=\"{current_description}\" "
#     f" --max_GPUs=1 --learning_rate=10e-6 --num_epochs=1 --patience=6 --num_serialized_models_to_keep=1 --flagfile=configs/base.txt"
#     f" --d_batch=8 --max_seq_length=256 "
#     f" --DIR=combo"
#     f" --replace_self_predictions='alternate'"
#     f" --use_HFpretrained_weights"
#     f" --DIR_loss_fraction={current_lambda}",
#
#     f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#     f' --pretrained_model={current_run_name} --max_GPUs=1 '
#     f' --overrides "run_name={current_run_name}"; cd ..'
# ],
#     'description': current_description,
#     'server': current_server}

# current_server = 'arwen'
# current_lambda = 0.3
# current_run_name = f"combo_CEloss_lambda_{current_lambda}_HFpre_WBG"
# current_description = f"Test effect of Tian-et-al-inspired loss, to be compared with normal lambda_{current_lambda}_HFpre_WBG"
# RUNS[current_run_name] = {'commands': [
#     f"ssh {current_server}",
#
#     f"conda activate p1;python pretrain.py --run_name={current_run_name} --description=\"{current_description}\" "
#     f" --max_GPUs=1 --learning_rate=10e-6 --num_epochs=1 --patience=6 --num_serialized_models_to_keep=1 --flagfile=configs/base.txt"
#     f" --d_batch=8 --max_seq_length=256 "
#     f" --DIR=combo"
#     f" --contrastive_loss=CE"
#     f" --use_HFpretrained_weights"
#     f" --DIR_loss_fraction={current_lambda}",
#
#     f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#     f' --pretrained_model={current_run_name} --max_GPUs=1 '
#     f' --overrides "run_name={current_run_name}"; cd ..'
# ],
#     'description': current_description,
#     'server': current_server}

# for current_server, current_lambda in zip(
#
#         ['bilbo', 'bilbo','bilbo'],
#         [0.4, .6,.9]
# ):
#     current_run_name = f"lambda_{current_lambda}_HFpretrain_WBG_r2"
#     current_description = "A second run (with bigger batch size) of: Updated to work with wiki+bc+gb data: pretraining with different fractions lambda of" \
#                           " DIR loss objective, seeing if improvement at any lambda." \
#                           "No DAR, only DAO "
#     RUNS[current_run_name] = {'commands': [
#         f"ssh {current_server}",
#
#         f"conda activate p1;python pretrain.py --run_name={current_run_name} --description=\"{current_description}\" "
#         f" --max_GPUs=1 --learning_rate=10e-6 --num_epochs=1 --patience=6 --num_serialized_models_to_keep=1 --flagfile=configs/base.txt"
#         f" --d_batch=8 --max_seq_length=256 "
#         f" --DIR=combo"
#         f" --replace_self_predictions=''"
#         f" --use_HFpretrained_weights"
#         f" --DIR_loss_fraction={current_lambda}",
#
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f' --pretrained_model={current_run_name} --max_GPUs=1 '
#         f' --overrides "run_name={current_run_name}"; cd ..'
#     ],
#         'description': current_description,
#         'server': current_server}

# current_server = 'frodo'
# current_lambda = 1
# current_source_run = "lambda_0.3_HFpretrain_WBG"
# current_run_name = f"retrained_self_pred_from_{current_source_run}"
# current_description = f"Test for slow features: see if a new self-predictor can more quickly get it right when " \
#                       f"representations have been influenced by a previous self-prediction objective (from {current_source_run})"
# RUNS[current_run_name] = {'commands': [
#     f"ssh {current_server}",
#
#     f"conda activate p1;python pretrain.py --run_name={current_run_name} --description=\"{current_description}\" "
#     f" --max_GPUs=1 --learning_rate=10e-6 --num_epochs=1 --patience=6 --num_serialized_models_to_keep=1 --flagfile=configs/base.txt"
#     f" --d_batch=8 --max_seq_length=256 "
#     f" --DIR=combo"
#     f" --freeze_main_model"
#     f" --retrain_self_predictor"
#     f" --selfpretrained_weights_path=/cw/working-frodo/nathan/phd/output/pretraining/{current_source_run}/best.th"
#     f" --DIR_loss_fraction={current_lambda}",
#
#     # f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#     # f' --pretrained_model={current_run_name} --max_GPUs=1 '
#     # f' --overrides "run_name={current_run_name}"; cd ..'
# ],
#     'description': current_description,
#     'server': current_server}
#
# current_server = 'rose'
# current_lambda = 1
# current_source_run = "vanilla"
# current_run_name = f"retrained_self_pred_from_{current_source_run}"
# current_description = f"Test for slow features: see if a new self-predictor can less quickly get it right when " \
#                       f"representations haven't been influenced by a previous self-prediction objective (aka when building on {current_source_run})"
# RUNS[current_run_name] = {'commands': [
#     f"ssh {current_server}",
#
#     f"conda activate p1;python pretrain.py --run_name={current_run_name} --description=\"{current_description}\" "
#     f" --max_GPUs=1 --learning_rate=10e-6 --num_epochs=1 --patience=6 --num_serialized_models_to_keep=1 --flagfile=configs/base.txt"
#     f" --d_batch=8 --max_seq_length=256 "
#     f" --DIR=combo"
#     f" --freeze_main_model"
#     f" --retrain_self_predictor"
#     f" --use_HFpretrained_weights"
#     f" --DIR_loss_fraction={current_lambda}",
#
#     # f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#     # f' --pretrained_model={current_run_name} --max_GPUs=1 '
#     # f' --overrides "run_name={current_run_name}"; cd ..'
# ],
#     'description': current_description,
#     'server': current_server}

# current_server = 'rose'
# current_lambda = 0.3
# current_run_name = f"combo_small_pred_lambda_{current_lambda}_HFpre_WBG_v2"
# current_description = "Testing whether a smaller predictor (=better for mem) can improve over vanilla / " \
#                       "over bigger self-predictor. This time without 'alternating' switch, so I can compare to " \
#                       "lambda_0.3_HFpretrain_WBG_r2"
# RUNS[current_run_name] = {'commands': [
#     f"ssh {current_server}",
#
#     f"conda activate p1;python pretrain.py --run_name={current_run_name} --description=\"{current_description}\" "
#     f" --max_GPUs=1 --learning_rate=10e-6 --num_epochs=1 --patience=6 --num_serialized_models_to_keep=1 --flagfile=configs/base.txt"
#     f" --d_batch=8 --max_seq_length=256 "
#     f" --DIR=combo"
#     f" --DIR_size=small"
#     f" --use_HFpretrained_weights"
#     f" --DIR_loss_fraction={current_lambda}",
#
#     f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#     f' --pretrained_model={current_run_name} --max_GPUs=1 '
#     f' --overrides "run_name={current_run_name}"; cd ..'
# ],
#     'description': current_description,
#     'server': current_server}

# for current_server, current_lambda in zip(
#
#         ['rose','frodo','frodo'],
#         [0,0.3,0.4]
# ):
#     current_run_name = f"lambda_{current_lambda}_HFpretrain_WBG_run3"
#     current_description = f"A third run at lambda {current_lambda}"
#     RUNS[current_run_name] = {'commands': [
#         f"ssh {current_server}",
#
#         f"conda activate p1;python pretrain.py --run_name={current_run_name} --description=\"{current_description}\" "
#         f" --max_GPUs=1 --learning_rate=10e-6 --num_epochs=1 --patience=6 --num_serialized_models_to_keep=1 --flagfile=configs/base.txt"
#         f" --d_batch=8 --max_seq_length=256 "
#         f" --DIR=combo"
#         f" --replace_self_predictions=''"
#         f" --use_HFpretrained_weights"
#         f" --DIR_loss_fraction={current_lambda}",
#
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f' --pretrained_model={current_run_name} --max_GPUs=1 '
#         f' --overrides "run_name={current_run_name}"; cd ..'
#     ],
#         'description': current_description,
#         'server': current_server}

# for current_server, current_lambda in zip(
#
#         ['sauron','rose'],
#         [0.6,0.9]
# ):
#     current_run_name = f"lambda_{current_lambda}_HFpretrain_WBG_run3"
#     current_description = f"A third run at lambda {current_lambda}"
#     RUNS[current_run_name] = {'commands': [
#         f"ssh {current_server}",
#
#         f"conda activate p1;python pretrain.py --run_name={current_run_name} --description=\"{current_description}\" "
#         f" --max_GPUs=1 --learning_rate=10e-6 --num_epochs=1 --patience=6 --num_serialized_models_to_keep=1 --flagfile=configs/base.txt"
#         f" --d_batch=8 --max_seq_length=256 "
#         f" --DIR=combo"
#         f" --replace_self_predictions=''"
#         f" --use_HFpretrained_weights"
#         f" --DIR_loss_fraction={current_lambda}",
#
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f' --pretrained_model={current_run_name} --max_GPUs=1 '
#         f' --overrides "run_name={current_run_name}"; cd ..'
#     ],
#         'description': current_description,
#         'server': current_server}

# for current_server, current_lambda, current_ablation in zip(
#         [
#             # 'sauron',
#             'arwen'
#         ],
#         [
#             # 0.4,
#             0.4
#         ],
#     [
#         # 'only_adjacent',
#         'only_top_down'
#     ]
# ):
#     current_run_name = f"HFpre_MLM_SOP_lambda_{current_lambda}_{current_ablation}"
#     current_description = f"Run to compare with HFpre_MLM_SOP_lambda_{current_lambda}. " \
#                           f"Checks the impact of using {' '.join(current_ablation.split('_'))} states"
#     RUNS[current_run_name] = {'commands': [
#         f"ssh {current_server}",
#
#         f"conda activate p1;python pretrain.py --run_name={current_run_name} --description=\"{current_description}\" "
#         f" --max_GPUs=1 --learning_rate=10e-6 --num_epochs=1 --patience=6 --num_serialized_models_to_keep=1 --flagfile=configs/base.txt"
#         f" --d_batch=8 --max_seq_length=256 "
#         f" --DIR={current_ablation}"
#         f" --objective=albert_mlm_sop"
#         f" --replace_self_predictions=''"
#         f" --use_HFpretrained_weights"
#         f" --DIR_loss_fraction={current_lambda}",
#
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f' --pretrained_model={current_run_name} --max_GPUs=1 '
#         f' --overrides "run_name={current_run_name}"; cd ..'
#     ],
#         'description': current_description,
#         'server': current_server}

for current_server, current_lambda in zip(

        ['sauron','sauron']
            ,
        [0.9,1]
):
    current_run_name = f"HFpre_MLM_SOP_lambda_{current_lambda}_r2"
    current_description = f"Run to compare with lambda_{current_lambda}_HFpretrain_WBG. Checks the impact of including " \
                          f"a proper Albert objective: SOP and proper MLM, for pairwise segmented input"
    RUNS[current_run_name] = {'commands': [
        f"ssh {current_server}",

        f"conda activate p1;python pretrain.py --run_name={current_run_name} --description=\"{current_description}\" "
        f" --max_GPUs=1 --learning_rate=10e-6 --num_epochs=1 --patience=6 --num_serialized_models_to_keep=1 --flagfile=configs/base.txt"
        f" --d_batch=8 --max_seq_length=256 "
        f" --DIR=combo"
        f" --objective=albert_mlm_sop"
        f" --replace_self_predictions=''"
        f" --use_HFpretrained_weights"
        f" --DIR_loss_fraction={current_lambda}",

        f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
        f' --pretrained_model={current_run_name} --max_GPUs=1 '
        f' --overrides "run_name={current_run_name}"; cd ..'
    ],
        'description': current_description,
        'server': current_server}


def track_run_in_sheets(run_name, commands, description, server):
    SPREADSHEET_ID = '1JBFTrsLGd35ZZ2ATbOmv6WzF57A2xtEhneU5vQQXtj4'
    SHEET_ID = 0
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'google_sheets_credentials.json', SCOPES)
            creds = flow.run_console()
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)

    # The ID of the spreadsheet to update.
    spreadsheet_id = SPREADSHEET_ID

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    repo = git.Repo(search_parent_directories=True)
    git_sha = repo.head.object.hexsha
    git_link = f"https://github.com/Natithan/phd/commit/{git_sha}"
    script = ';'.join(commands)
    status = 'Running'
    joiner = "__JOINER__"

    batch_update_values_request_body = {
        'requests': [
            {
                "insertDimension": {
                    "range": {
                        "dimension": "ROWS",
                        "startIndex": 1,
                        "endIndex": 2,
                        "sheetId": SHEET_ID
                    },
                    "inheritFromBefore": False
                }
            },
            {
                "pasteData": {
                    "coordinate": {
                        "rowIndex": 1,
                        "columnIndex": 0,
                        "sheetId": SHEET_ID
                    },

                    "data": joiner.join([
                        timestamp, run_name, description, git_sha, git_link, server, script, status
                    ]),
                    "delimiter": joiner
                }
            }
        ]
    }

    request = service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=batch_update_values_request_body)

    response = request.execute()

    pprint(response)


server = libtmux.Server()
session = server.find_where({"session_name": "exps"})
assert session is not None, "Don't forget to start a tmux session"
for run_name, run_values in RUNS.items():
    commands = run_values['commands']
    description = run_values['description']
    server = run_values['server']
    ws = [w for w in session.windows if w['window_name'] == run_name]
    assert len(ws) < 2, f"Found two or more windows with name {run_name}"
    if len(ws) == 0:
        w = session.new_window(attach=False, window_name=run_name)
    else:
        w = ws[0]
    pane = w.panes[0]
    print(f"Sending following commands to window  {w['window_name']} : {';'.join(commands)}")
    print("Logging run in google sheet")
    track_run_in_sheets(run_name, commands, description, server)
    for command in commands:
        if 'ssh' in command:
            pane.send_keys('hostname')
            time.sleep(1)  # Wait for the output to be printed
            current_host = pane.cmd('capture-pane', '-p').stdout[-2]
            if command == f'ssh {current_host}':  # Don't ssh extra to a host we're already on
                continue
            else:
                pane.send_keys(command)
                pane.send_keys('screen')
        else:
            pane.send_keys(command)
    time.sleep(20)  # To make sure the same GPUs aren't picked
