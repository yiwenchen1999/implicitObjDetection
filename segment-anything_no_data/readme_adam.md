ssh aren10@ssh.ccv.brown.edu
interact -n 1 -t 12:00:00 -m 32g -q gpu  -X -g 1 -f geforce3090
module load cuda/11.1.1
module load cudnn/8.2.0
module load anaconda/2020.02
source /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate PL3DS_Baseline

scp -r /Users/jfgvl1187/Desktop/segment-anything aren10@ssh.ccv.brown.edu:/users/aren10/segment-anything


scp -r /Users/jfgvl1187/Desktop/segment-anything/main.py aren10@ssh.ccv.brown.edu:/users/aren10/segment-anything/main.py


scp -r aren10@ssh.ccv.brown.edu:/users/aren10/segment-anything/output_data/r_16_clip2d_gt.pt /Users/jfgvl1187/Desktop/segment-anything/output_data/r_16_clip2d_gt.pt
______

scp -r /Users/jfgvl1187/Desktop/segment-anything/segment_anything/Reproject_CLIP/train_test.py aren10@ssh.ccv.brown.edu:/users/aren10/segment-anything/segment_anything/Reproject_CLIP/train_test.py

scp -r /Users/jfgvl1187/Desktop/segment-anything/segment_anything/Reproject_CLIP/config.py aren10@ssh.ccv.brown.edu:/users/aren10/segment-anything/segment_anything/Reproject_CLIP/config.py

scp -r /Users/jfgvl1187/Desktop/segment-anything/segment_anything/Reproject_CLIP/modules.py aren10@ssh.ccv.brown.edu:/users/aren10/segment-anything/segment_anything/Reproject_CLIP/modules.py

scp -r aren10@ssh.ccv.brown.edu:/users/aren10/segment-anything/segment_anything/Reproject_CLIP/best.pt /Users/jfgvl1187/Desktop/segment-anything/segment_anything/Reproject_CLIP/best.pt