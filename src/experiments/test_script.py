import libtmux
from config import FLAGS

MINI_CHECK = False
RUNS = {}
# current_run_name = "HFRoberta_HFpre_nomypre"
# RUNS[current_run_name] = [
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f'--max_GPUs=1 '
#         f'--overrides "run_name={current_run_name},'
#         f'input_module=roberta-base'
#         f'"; cd ..'
#     ]
current_run_name = "HFAlbert_HFpre_nomypre"
RUNS[current_run_name] = [
        f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
        f'--max_GPUs=1 '
        f'--overrides "run_name={current_run_name},'
        f'input_module={FLAGS.hf_model_handle}'
        f'"; cd ..'
    ]
#
# current_run_name = "baseline_HFpre_nomypre_2"
# RUNS[current_run_name] = [
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f" --use_pretrained_weights "
#         f'--max_GPUs=1 '
#         f'--overrides "run_name={current_run_name},'
#         f'input_module=dirt'
#         f'"; cd ..'
#     ]


# current_run_name = "baseline_HFpre_mypre"
# RUNS[current_run_name] = [
#         f"python pretrain.py --max_GPUs=1 --d_batch=2 "
#         f" --use_pretrained_weights "
#         f" --run_name={current_run_name}"
#         f" --description='HFpretrained my baseline WITH mypretrain -> check vs my baseline with no mypretrain, form baseline for DIRT alts'",
#
#         f'cd jiant; conda activate jiant; python my_main.py --config_file jiant/config/superglue_dirt.conf '
#         f' --pretrained_model={current_run_name} --max_GPUs=1 '
#         f' --overrides "run_name={current_run_name},input_module=dirt"; cd ..'
#     ]
# current_run_name = "baseline_noHFpre_mypre"
# RUNS[current_run_name] = [
#         f"python pretrain.py --max_GPUs=1 --d_batch=2 "
#         f" --run_name={current_run_name}"
#         f" --description='From scratch my Albert with mypretrain -> check if here also ok vs HF Albert + form baseline for DIRT alts, aiming-for-relative-improvements'",
#
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
# current_run_name = "top_down_noHFpre_mypre"
# RUNS[current_run_name] = [
#         f"python pretrain.py --max_GPUs=1 --d_batch=2 "
#         f" --run_name={current_run_name}"
#         f"--DIR=top_down"
#         f"--description='No HFpretrain my preffered DIRT alt (aka top_down) with mypretrain -> check if improvement somewhere vs my albert, aiming-for-relative-improvements'",
#
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


server = libtmux.Server()
session = server.list_sessions()[0]
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
