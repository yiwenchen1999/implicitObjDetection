# To run on the Brown University Cluster 
Run `bash scripts/load_env.sh`

`module load cuda/10.2`
`module load cudnn/7.6.5`
`module load anaconda/2020.02` 
`conda create -n $name_of_your_choice`
`source activate $name_of_your_choice`
 
Then proceed with installing the dependencies on https://github.com/yiwenchen1999/implicitObjDetection#dependencies


# Download the dataset on the cluster 
You can download the dataset by using the following command, but before you download check your quota first as the dataset is approximately 55GB - you have been warned!

`myquota` on the terminal will help you check what directory is free `~/scratch` should have a lot of space. 
 
`wget https://storage.googleapis.com/kubric-public/data/NeSFDatasets/NeSF%20datasets/toybox-13.tar.gz`

To unzip: 
`tar -xvf toybox-13.tar.gz`

# Running a jupyter notebook 
Run `bash scripts/start_notebook.sh`
Take the printed ssh command and run it on a local terminal (this is not the oscar terminal but the one locally on your computer).
If the token does not show up it should in the whatever directory you ran the start_notebook script under `jupyter-log-xxxxxxx.txt` 


