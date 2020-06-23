import git
import subprocess

# import os
# print(os.getcwd())
# import paramiko
# hostname = 'arwen.cs.kuleuven.be'
# myuser   = 'nathan'
# mySSHK   = '/path/to/sshkey.pub'
# sshcon   = paramiko.SSHClient()  # will create the object
# sshcon.set_missing_host_key_policy(paramiko.AutoAddPolicy()) # no known_hosts error
# sshcon.connect(hostname, username=myuser, key_filename=mySSHK) # no passwd needed
# ssh = paramiko.SSHClient()
# ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#
# ssh.connect('arwen.cs.kuleuven.be', username='nathan', password=password)
# ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(cmd_to_execute)
# HOST="arwen"
# # Ports are handled in ~/.ssh/config since we use OpenSSH
# COMMAND="python experiments/tmux_run_launcher.py"
# import os
#
# s = subprocess.run(["wsl", "ssh","arwen"],input='python experiments/git_launcher_wrapper.py',text=True)

COMMIT_MESSAGE = 'Automatic commit for experiment'

def git_push():
    try:
        repo = git.Repo(search_parent_directories=True)
        repo.git.add(update=True)
        repo.index.commit(COMMIT_MESSAGE)
        origin = repo.remote(name='origin')
        origin.push()
    except:
        print('Some error occured while pushing the code')

git_push()
s = subprocess.run(["wsl", "ssh","arwen"],input='cd experiments;python tmux_run_launcher.py',text=True)

