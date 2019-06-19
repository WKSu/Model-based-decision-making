import subprocess
import time

cmd = []
cmd.append("git add .")
cmd.append('git commit -m "Test auto git push - timed"')
cmd.append("git push")

for c in cmd:
    returned_output = subprocess.call(cmd, shell=True)  # returns the exit code in unix
    print('returned output:', returned_output)
    time.sleep(3)
