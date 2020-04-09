import subprocess

MASKING_TOKEN = '@@mask@@'
EOS_TOKEN = '@@MASK_EOS@@'
TO_BE_DELETED_TOKEN = '@@TO_BE_DELETED@@'
DECODER_START_TOKEN = '@@START@@'
BPE_INDEXER_SUFFIX = '</w>'
READ_ONLY_ROOT = '/cw/working-arwen/nathan/phd'
HOSTNAME = subprocess.check_output('hostname').decode().strip()
WRITE_ROOT = f'/cw/working-{HOSTNAME}/nathan/phd'
HF_MODEL_HANDLE = 'albert-xlarge-v1' #v2 seems to be faulty: https://github.com/huggingface/transformers/pull/1683#issuecomment-556001607
TYPE_VOCAB_SIZE = 2