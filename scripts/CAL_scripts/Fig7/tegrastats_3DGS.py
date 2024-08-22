import subprocess
import time
import os
import sys

if len(sys.argv) != 2:
    print("Usage: python script_name.py <model_name>")
    sys.exit(1)

model_name = sys.argv[1]

# Command to start 3DGS rendering
render_command = ["python3", "/home/jetson-agx/NeRF/gaussian-splatting/render.py", "-m", f"/home/jetson-agx/NeRF/models_pretrained/{model_name}_30k", "--skip_train"]

# Command to start tegrastats
tegrastats_command = ["sudo", "tegrastats", "--verbose", "--interval", "50", "--logfile", f"/home/jetson-agx/NeRF/CAL_3DGS_rev/fig7_XavierAGX_20240822/{model_name}.csv"]

# Start the 3DGS rendering process
render_process = subprocess.Popen(render_command)

# Start tegrastats
tegrastats_process = subprocess.Popen(tegrastats_command)

# Wait a moment to ensure tegrastats starts
time.sleep(1)

# Find the actual PID of the tegrastats process
try:
    sudo_pids_output = subprocess.check_output(["pgrep", "-f", "sudo tegrastats --verbose"]).strip().decode()
    sudo_pids = sudo_pids_output.split("\n")
    for sudo_pid in sudo_pids:
        sudo_pid = sudo_pid.strip()
        if sudo_pid.isdigit():
            sudo_pid = int(sudo_pid)
            # List child processes of the sudo process
            child_pids_output = subprocess.check_output(["pgrep", "-P", str(sudo_pid)]).strip().decode()
            child_pids = child_pids_output.split("\n")
            for child_pid in child_pids:
                child_pid = child_pid.strip()
                if child_pid.isdigit():
                    child_pid = int(child_pid)
                    # Verify the command line of the child process to ensure it is the correct one
                    cmdline = subprocess.check_output(["ps", "-p", str(child_pid), "-o", "cmd="]).strip().decode()
                    if "tegrastats --verbose" in cmdline:
                        tegrastats_pid = child_pid
                        print(f"Verified tegrastats PID: {tegrastats_pid}")
                        break
            else:
                continue
            break
    else:
        tegrastats_pid = None
        print("No matching tegrastats process found")
except subprocess.CalledProcessError:
    print("Failed to get tegrastats PID")
    tegrastats_pid = None

# Wait for the rendering process to complete
render_process.wait()

# Stop the tegrastats process after rendering is done
if tegrastats_pid:
    os.system(f"sudo kill {tegrastats_pid}")
else:
    print("tegrastats process not found")
