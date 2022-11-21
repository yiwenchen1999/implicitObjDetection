#!/bin/bash
# Script to queue a JupyterLab job and print the information about the job
# Removes and ends old jobs that might still be running
echo "Cleaning up (removing running jobs and cleaning up directory)"
for JOB in $(find -type f -name 'jupyter-log-*.txt')
do
    JOBID=${JOB:14:-4}
    scancel $JOBID
    rm $JOB
done

# Submits batch job to Slurm
sbatch ./scripts/submit_jupyterlab_to_queue.sh

# Waits, then prints information about the submitted job
while [ -z "$joblog" ]
do
    joblog=$(find -type f -name 'jupyter-log-*.txt')
    for JOB in $(find -type f -name 'jupyter-log-*.txt')
    do
        cat $JOB | head -n 22
        url=$(grep -o "localhost:[[:digit:]]*" $JOB)
        token=$()
        while [ -z "$token" ]
        do
            token=$(grep -o "/lab?token=[[:digit:][:lower:]]*$" $JOB | head -n 1)
            sleep 1
        done
        echo "URL at http://$url$token"
    done
    sleep 1
done
