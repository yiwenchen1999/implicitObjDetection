#!/bin/bash
#SBATCH --nodes 1
#SBATCH -c 6
#SBATCH --time 08:00:00
#SBATCH --mem 80G
#SBATCH --job-name jupyter-lab-tunnel
#SBATCH --output jupyter-log-%J.txt
## get tunneling info
XDG_RUNTIME_DIR=""
ipnport=$(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)
## print tunneling instructions to jupyter-log-{jobid}.txt
echo -e "
    Copy/Paste this in your local terminal to ssh tunnel with remote
    -----------------------------------------------------------------
    \033[31;1;4mssh -N -L $ipnport:$ipnip:$ipnport $USER@ssh.ccv.brown.edu\033[0m
    -----------------------------------------------------------------
    or if you are on Brown wifi and have SSH keys set up: 
    -----------------------------------------------------------------
    \033[31;1;4mssh -N -L $ipnport:$ipnip:$ipnport $USER@sshcampus.ccv.brown.edu\033[0m
    -----------------------------------------------------------------
    Then open a browser on your local machine to the following address
    ------------------------------------------------------------------
    localhost:$ipnport  (prefix w/ https:// if using password)
    ------------------------------------------------------------------
    "
## start an ipcluster instance and launch jupyter server
jupyter-lab --no-browser --port=$ipnport --ip=$ipnip
