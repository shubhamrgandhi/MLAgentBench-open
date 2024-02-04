import os
import subprocess
import selectors
import shutil
import glob
import sys
import inspect
from functools import wraps
import time
from io import StringIO
from .schema import Step, ActionInfo, Action, EnvException



script_path = 'train.py'
# device = kwargs["device"]
# python = kwargs["python"]
cmd = f"python -u {script_path}"
print("Executing script now...")
process = subprocess.Popen(
    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True, cwd='.'
)

# Read the output from the subprocess as it becomes available
stdout_lines = []
stderr_lines = []

while True:
    output_line = process.stdout.readline()
    if output_line == "" and process.poll() is not None:
        break
    print(output_line.strip())
    stdout_lines.append(output_line)

process.wait()
process.terminate()

for line in process.stderr:
    line = line
    print("STDERR:", line, end =" ")
    stderr_lines.append(line)

return_code = process.returncode

if return_code != 0:
    observation = "".join(stderr_lines)
else:
    observation = "".join(stdout_lines)
if observation == "" and return_code == 0:
    # printed to stderr only
    observation = "".join(stderr_lines)

print("The script has been executed. Here is the output:\n" + observation)