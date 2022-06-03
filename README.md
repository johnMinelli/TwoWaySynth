# TwoWaySynth 

## Prerequisite

```bash
pip3 install -r requirements.txt
```

or install manually the following packages :

```
torch >= 1.8.1
torchvision >= 0.9.1
pandas
matplotlib
scipy
scikit-image
imageio
argparse
wandb
tensorboardX
dominate
progressbar2
termcolor
path
pebble
tqdm
```

## Preparing training data
Download ShapeNet dataset and run the following command

For [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php), first download the dataset using this [script](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) provided on the official website, and then run the following command. The `--depth` option will save resized copies of groundtruth depth. The `--with-pose` will dump the sequence pose
```bash
python3 preprocess\prepare_data.py --data_path $DATA$/datasets/ShapeNet --dataset shapenet --dataset_format shapenet --height 256 --width 256 --dump_root $DATA$/datasets/ShapeNet_formatted --num_threads 1 --depth sparse --with_pose
```

## Training
Once the data are formatted following the above instructions, you should be able to train the model by running the following command
```bash
python3 train.py --name shapenet --dataset shapenet --shuffle_batches --validate --data_path $DATA$/datasets/ShapeNet_formatted --depth sparse --train_file ./datasets/shapenet_chair_split/id_train.txt --test_file ./datasets/shapenet_chair_split/id_test.txt
```
Additionally you can visualize metrics on Weight and Bias adding `--wandb` or setup tensorboard `--tensorbaord` and starting it
```bash
tensorboard --logdir=save/
```
and visualize the training progress by opening [https://localhost:8097](https://localhost:8097) on your browser starting the visdom server locally
```bash
visdom
```
You can customize visualization with  `--display_freq 1 --display_port 8097`

## Evaluation
Use a pretrained model to run the evaluation on images pairs 
```bash
python3 eval.py --name shapenet --dataset shapenet --save_images --save_path ./save --models_path ./save/shapenet --data_path $DATA$/datasets/ShapeNet_formatted --test_file ./datasets/shapenet_chair_split/eval_pairs_40.txt --model_epoch 30
```
## Pretrained Models

[//]: # ([Chairs]&#40;https://drive.google.com/drive/folders/&#41;)
[//]: # ([Cars]&#40;https://drive.google.com/drive/folders/&#41;)