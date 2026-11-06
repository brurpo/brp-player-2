import subprocess
import sys

# Path to the Python executable (e.g., 'python', 'python3', or a full path)
python_executable = sys.executable

# Path to the script you want to execute
script_to_run = "./player.py"

# Arguments to pass to the script (optional)
script_args = ["arg1"]

try:
    # Execute the script using subprocess.run()
    # capture_output=True captures stdout and stderr
    # text=True decodes stdout/stderr as text
    result = subprocess.run(
        [python_executable, script_to_run] + script_args,
        capture_output=True,
        text=True,
        check=True  # Raises CalledProcessError if the script returns a non-zero exit code
    )

    print("Script output (stdout):")
    print(result.stdout)
    print("Script errors (stderr):")
    print(result.stderr)
    print(f"Script exited with code: {result.returncode}")

except subprocess.CalledProcessError as e:
    print(f"Error executing script: {e}")
    print(f"Stderr: {e.stderr}")
except FileNotFoundError:
    print(f"Error: Python executable or script '{script_to_run}' not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
