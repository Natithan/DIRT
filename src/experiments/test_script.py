import libtmux

SESSION_NAME = "experiments"
RUNS = [[
"python pipeline.py --use_pretrained_weights --run_name=baseline_HFpre_mypre --max_GPUs=1 --d_batch=2 "
"--description='HFpretrained my baseline WITH mypretrain -> check vs my baseline with no mypretrain, form baseline for DIRT alts'",

]]

server = libtmux.Server()
session = server.find_where({"session_name": SESSION_NAME})
w = session.new_window(attach=False, window_name="baseline_HFpre_mypre")
for run in RUNS:
    w = session.new_window(attach=False, window_name="baseline_HFpre_mypre")
    pane = w.panes[0]
    for command in run:
        pane.send_keys(command)
