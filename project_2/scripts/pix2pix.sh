echo Automatically Generating Pix2Pix Pics

rm pix2pix_out/*.png

python ../../pix2pix-tensorflow/tools/process.py --input_dir edges --output_dir temp --operation combine --b_dir edges --use_cpu

python ../../pix2pix-tensorflow/pix2pix.py --input_dir temp --output_dir pix2pix_out --mode test --checkpoint ../../pix2pix-tensorflow/faces_train --use_cpu

rm -rf temp
rm -rf edges
mkdir edges

rm pix2pix_out/events*
rm pix2pix_out/images/*inputs.png
rm pix2pix_out/images/*targets.png
rm pix2pix_out/graph.pbtxt
rm pix2pix_out/*.html
rm pix2pix_out/options.json

mv pix2pix_out/images/* pix2pix_out/
rmdir pix2pix_out/images
