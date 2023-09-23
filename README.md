# Depth self-supervision for single image novel view synthesis 

Merging Single Image Depth Estimation task with Novel View Synthesis in a single pipeline self supervised

## Prerequisite

```bash
pip3 install -r requirements.txt
```
You will also need the [Direct Warp module](https://github.com/ClementPinard/direct-warper)

## Preparing training data
ShapeNet dataset is already preformatted, you can download it from: [Google Drive](https://drive.google.com/file/d/1tpgl4Ts1TTYmD6gj-VdNa_4_8JAMK-N-/view?usp=sharing)

For [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php), first download the dataset using this [script](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) from the official website, and [depth data](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip), then run the following command. 
```bash
python3 preprocess\prepare_data.py --data_path $DATA$/datasets/ShapeNet --dataset kitti --height 256 --width 256 --dump_root $DATA$/datasets/ShapeNet_formatted --static_frames preprocess/static_frames.txt --num_threads 1 --depth sparse --with_pose
```
The `--depth` option will save resized copies of ground truth depth. The `--with-pose` will dump the sequence pose

## Training
Once the data are formatted following the above instructions, you should be able to train the model by running the following command
```bash
[ShapeNet] python3 train.py --dataset shapenet --data_path $DATA$/datasets/ShapeNet_formatted --train_file ./datasets/shapenet_chair_split/id_train.txt --valid_file ./datasets/shapenet_chair_split/id_valid.txt \
 --batch_size 8 --lambda_recon 100 --lambda_warp 100 --lambda_vgg 100 --lambda_consistency 100 --lambda_smooth 10 --epochs 25 

[KITTI] python3 train.py --dataset kitti --data_path $DATA$/datasets/KITTI_formatted --train_file ./datasets/kitti_split/eigen_train_files.txt --valid_file ./datasets/kitti_split/eigen_val_files.txt \
 --batch_size 8 --lambda_recon 100 --lambda_warp 100 --lambda_vgg 100 --lambda_consistency 25 --lambda_smooth 25 --epochs 25 
```

Additionally, you can visualize metrics on Weight and Bias adding `--wandb` or setup tensorboard `--tensorbaord` and starting it
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
[ShapeNet] python3 eval.py --dataset shapenet --save_images --save_path ./save --models_path ./save/shapenet/chair --data_path $DATA$/datasets/ShapeNet_formatted --test_file ./datasets/shapenet_chair_split/eval_pairs_40.txt --model_epoch -1

[KITTI] python3 eval.py --dataset kitti --save_images --save_path ./save --models_path ./save/kitti --data_path $DATA$/datasets/ShapeNet_formatted --test_file ./datasets/kitti_split/eigen_test_files.txt --model_epoch -1
```
In-painting (alternative to the above script) and dense depth rendering for KITTI depth evaluation has been computed with: https://github.com/wangq95/KITTI_Dense_Depth

<!-- ## Pretrained Models -->

[//]: # ([Chairs]&#40;https://drive.google.com/drive/folders/&#41;)
[//]: # ([Cars]&#40;https://drive.google.com/drive/folders/&#41;)
[//]: # ([KITTI]&#40;https://drive.google.com/drive/folders/&#41;)

## Cite
```
@misc{minelli2023depth,
      title={Depth self-supervision for single image novel view synthesis}, 
      author={Giovanni Minelli and Matteo Poggi and Samuele Salti},
      year={2023},
      eprint={2308.14108},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
