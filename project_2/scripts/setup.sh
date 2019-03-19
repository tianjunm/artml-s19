cd ~
sudo pip install pycocotools

git clone https://github.com/matterport/Mask_RCNN
cd Mask_RCNN
sudo pip install -r requirements.txt
python setup.py install

cd ~
git clone https://github.com/affinelayer/pix2pix-tensorflow

mv pix2pix.sh Mask_RCNN/samples/
mv demo.ipynb Mask_RCNN/samples/
mv pix2pix.py pix2pix-tensorflow/
mv process.py pix2pix-tensorflow/tools/
