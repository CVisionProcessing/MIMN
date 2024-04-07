# MIMN
Multi-Involution Memory Network for Unsupervised Video Object Segmentation

## Download Datasets
In the paper, we use the following two public available dataset for training. Here are some steps to prepare the data:

[DAVIS 2016: TrainVal-Images and Annotations](https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip)

[DUTS-TR: Training (images and ground-truth)](http://saliencydetection.net/duts/download/DUTS-TR.zip)

## Prepare Optical Flow
Please follow [raft](https://github.com/princeton-vl/RAFT?tab=readme-ov-file#raft) to generate optical flow maps and put into *"dataset"* folder. 

## Train
### stage 1
1\. Check datasets path in *"args.py"*  and *"dataset.py"* file.

2\. Run *"train_backbone.py"*
```
python train_backbone.py --model_name MIMN_backbone
```

### stage 2
1\. Check datasets path in *"args.py"* and *"dataset_pro.py"* file.

2\. Run *"train.py"*
```
python train.py --model_name MIMN --checkpoint MIMN_backbone --ckptEpoch 39
```
If you want to continue training from a certain epoch:
```
python train.py --model_name MIMN --resume --epoch_resume 89
```

Other initialisation details are in *"args.py"* file.

## Test
1\. Make sure the pre-trained models are in your *"ckpt"* folder.

2\. Select a pre-trained model and testing datasets in *"test.py"* file.

3\. Run *"test.py"*
```
python test.py --model_name MIMN --epoch 89 --dataset DAVIS
```

4\. Apply CRF post-processing:
Please follow the [Installation](https://github.com/lucasb-eyer/pydensecrf?tab=readme-ov-file#installation) section to install ```pydensecrf```. 
```
cd misc

python apply_densecrf.py
```
