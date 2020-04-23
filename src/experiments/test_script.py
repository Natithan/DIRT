import time

import libtmux
from config import FLAGS

MINI_CHECK = False
RUNS = {}
current_run_name = "HFRoberta_HFpre_nomypre_2"
current_description = "Re-running the roberta baseline to (hopefully) have the results be stored in my results sheet, " \
                      "including the total average score on validation data. " \
                      "That average can then be compared to leaderboard. (this is pure roberta directly training on target SG tasks)"
RUNS[current_run_name] = [
        f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
        f' --max_GPUs=1 '
        f' --description="{current_description}"'
        f' --overrides "'
        f' run_name={current_run_name},'
        f' input_module=roberta-base'
        f'"; cd ..'
    ]

current_run_name = "HFAlbert_HFpre_nomypre_2"
current_description = "Re-running the albert baseline (in it's small version that I use) " \
                      "to (hopefully) have the results be stored in my results sheet, including the total average score." \
                      "That average can then be compared to roberta baseline"

RUNS[current_run_name] = [
        f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
        f' --max_GPUs=1 '
        f' --description="{current_description}"'
        f' --overrides "'
        f' run_name={current_run_name},'
        f' input_module={FLAGS.hf_model_handle}'
        f'"; cd ..'
    ]


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

current_run_name = "combo_noHFpre_mypre"
RUNS[current_run_name] = [
        f"python pretrain.py --max_GPUs=1 --d_batch=3"
        f" --run_name={current_run_name}"
        f" --DIR=combo"
        f" --d_hidden=768"
        f" --d_ff=3072"
        f" --d_top_down=3072"
        f" --nb_heads=12"
        f" --nb_encoder_layers=12"
        f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
        f'--pretrained_model={current_run_name} --max_GPUs=1 '
        f'--overrides "run_name={current_run_name},input_module=dirt"; cd ..'
    ]

current_run_name = "vanilla_noHFpre_mypre"
RUNS[current_run_name] = [
        f"python pretrain.py --max_GPUs=1 --d_batch=3"
        f" --run_name={current_run_name}"
        f" --d_hidden=768"
        f" --d_ff=3072"
        f" --d_top_down=3072"
        f" --nb_heads=12"
        f" --nb_encoder_layers=12"
        f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
        f'--pretrained_model={current_run_name} --max_GPUs=1 '
        f'--overrides "run_name={current_run_name},input_module=dirt"; cd ..'
    ]

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


server = libtmux.Server()
session = None
for s in server.list_sessions():
    if s['session_name'] != "tb":
        session = s
assert session is not None, "Don't forget to start a tmux session"
for run_name, commands in RUNS.items():
    if MINI_CHECK:
        run_name += "_mini"
    w = session.new_window(attach=False, window_name=run_name)
    pane = w.panes[0]
    for command in commands:
        if MINI_CHECK:
            command += ' --mini' # To train on mini subset of data
            command += ' --num_epochs=1'
            command += ' --overrides="max_epochs=1"'
            command += ' --hf_model_handle=albert-base' #smaller model TODO then make sure all model-size flags also adjust
        print(f"Sending command: {command}")
        print(f"To window: {w['window_name']}")
        pane.send_keys(command)
    time.sleep(10) # To make sure the same GPUs aren't picked
