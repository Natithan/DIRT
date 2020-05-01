import time

import libtmux
from pathlib2 import Path

from constants import HOSTNAME, WRITE_ROOT, READ_ONLY_ROOT

from config import FLAGS

RUNS = {}


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
#         f" --use_pretrained_weights "
#         f'--max_GPUs=1 '
#         f' --'
#         f'--overrides "run_name={current_run_name},'
#         f'input_module=dirt'
#         f'"; cd ..'
#     ]


# current_run_name = "baseline_HFpre_mypre"
# RUNS[current_run_name] = [
#         f"python pretrain.py --max_GPUs=1 --d_batch=2 "
#         f" --use_pretrained_weights "
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
#         f" --use_pretrained_weights "
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
#         f' --use_pretrained_weights'
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
        # f' --use_pretrained_weights'
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
current_server = 'bilbo'
current_run_name = "combo_HFpre_mypre"
current_description = "- See if pretrained weights help in doing self-regression    \r\n" \
                      "- See if doing extra training with DIRT objective on top of pretrained weights improves (any aspect of) GLUE performance"
RUNS[current_run_name] = [
        f"ssh {current_server}",

        f"python pretrain.py --max_GPUs=1 --d_batch=3 --patience=1"
        f" --run_name={current_run_name}"
        f' --description="{current_description}"'
        f' --use_pretrained_weights'
        f" --DIR=combo"
        f" --flagfile=configs/base.txt",


        f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
        f' --pretrained_model={current_run_name} --max_GPUs=1 '
        f' --overrides "run_name={current_run_name},input_module=dirt"; cd ..'
    ]

current_server = 'bilbo'
current_run_name = "combo_HFpre_mypre"
current_description = "- See if pretrained weights help in doing self-regression    \r\n" \
                      "- See if doing extra training with DIRT objective on top of pretrained weights improves (any aspect of) GLUE performance"
RUNS[current_run_name] = [
        f"ssh {current_server}",

        f"python pretrain.py --max_GPUs=1 --d_batch=3 --patience=1"
        f" --run_name={current_run_name}"
        f' --description="{current_description}"'
        f' --use_pretrained_weights'
        f" --DIR=combo"
        f" --flagfile=configs/base.txt",


        f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
        f' --pretrained_model={current_run_name} --max_GPUs=1 '
        f' --overrides "run_name={current_run_name},input_module=dirt"; cd ..'
    ]

current_run_name = "combo_fraction_.5"
current_description = "Check effect of relative importance of DIRT loss in pretraining task vs default .95"
current_server = 'frodo'

RUNS[current_run_name] = [
    f"ssh {current_server}",
    f"python pretrain.py --max_GPUs=1 --d_batch=3 --patience=1"
        f" --run_name={current_run_name}"
        f' --description="{current_description}"'
        f" --DIR=combo"
        f" --d_hidden=768"
        f" --learning_rate=10e-6"
        f" --d_ff=3072"
        f" --d_top_down=3072"
        f" --nb_heads=12"
        f" --nb_encoder_layers=12",
        f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
        f'--pretrained_model={current_run_name} --max_GPUs=1 '
        f'--overrides "run_name={current_run_name},input_module=dirt"; cd ..'
    ]

server = libtmux.Server()
session = server.find_where({"session_name":"exps"})
# for s in server.list_sessions():
#     if s['session_name'] != "tb":
#         session = s
assert session is not None, "Don't forget to start a tmux session"


def track_run_in_sheets(run_name,commands):
    pass #TODO


for run_name, commands in RUNS.items():
    w = session.new_window(attach=False, window_name=run_name)
    pane = w.panes[0]
    track_run_in_sheets(run_name,commands)
    for command in commands:
        if command == f'ssh {HOSTNAME}': # Don't ssh extra to a host we're already on
            continue
        print(f"Sending following command to window  {w['window_name']} : \n{command}")
        pane.send_keys(command)
    time.sleep(20) # To make sure the same GPUs aren't picked
