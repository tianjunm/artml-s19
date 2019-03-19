## Contents in the directory
- pix2pix.py: Modified version of the original pix2pix code. Added an option to use only CPU
- pix2pix.sh: Script automatically generating pix2pix pictures
- demo.ipynb: Modefied version of Mask-RCNN demo used to generate most of the results  
- process.py: Modified version of the processing script in the pix2pix code. Added an option to use only CPU and changed edge detection algorithm from HED to Canny. Removed Caffe dependency
- setup.sh: Bash script used to set up the environment on AWS

## How to reproduce the experiments
1. Upload all the contents in this directory to AWS
2. Give `setup.sh` permissions
```
chmod +x setup.sh
```
3. Execute `setup.sh`
```
./setup.sh
```
4. Have fun