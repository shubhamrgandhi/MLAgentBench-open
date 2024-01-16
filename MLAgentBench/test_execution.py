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
from schema import Step, ActionInfo, Action, EnvException
import pyreadline3 # This is needed to make sure that the input() function works properly

def execute_script(script_name, work_dir = "../workspace/cifar10", **kwargs):
    if not os.path.exists(os.path.join(work_dir,script_name)):
        raise EnvException(f"The file {script_name} does not exist.")
    try:
        script_path = script_name
        # device = kwargs["device"]
        # python = kwargs["python"]
        cmd = f"python -u {script_path}"
        print("Done 1")
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True, cwd=work_dir
        )
        stdout_lines = []
        # Read the output from the subprocess as it becomes available
        while True:
            output_line = process.stdout.readline()
            if output_line == "" and process.poll() is not None:
                break
            print(output_line.strip())
            stdout_lines.append(output_line)

        process.wait()
        process.terminate()

        stdout_lines = "".join(stdout_lines)
        stderr_lines = process.stderr.read()
        observation = stderr_lines if stderr_lines else stdout_lines

        return "The script has been executed. Here is the output:\n" + observation

    except Exception as e:
        raise EnvException(f"Something went wrong in executing {script_name}: {e}. Please check if it is ready to be executed.")

if __name__ == '__main__':
    obs = execute_script('train.py')
    print("\n\nOBSERVATION: \n\n", obs)