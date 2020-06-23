import git
import subprocess


# Logic behind this: First do the commit locally.
# Then make sure to sync the .git folder to remote, as the current commit is logged in the outputs sheet.
# Then run the tmux launcher on one of the always-on remotes.


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
subprocess.run(["wsl", 'rsync', '-chavzP', "/mnt/c/Users/natha/PycharmProjects/phd/.git", 'nathan@arwen.cs.kuleuven.be:/cw/working-arwen/nathan/phd/'])
subprocess.run(["wsl", "ssh","arwen"],input='cd experiments;python tmux_run_launcher.py',text=True)

