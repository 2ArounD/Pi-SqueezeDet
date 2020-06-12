## _SqueezeDet pruned for a Raspberry Pi:_ Accelerating neural networks embedded on a microcomputer trained on KITTI.
By Arnoud Jonker

This repository contains a tensorflow implementation of training and pruning SqueezeDet. Pruning is aimed for performance increase on a Raspberry Pi. Code for conversion of the network to tensorflow lite is included as well.

The repository is based on the original repository of squeezedet.

@inproceedings{squeezedet,
        Author = {Bichen Wu and Forrest Iandola and Peter H. Jin and Kurt Keutzer},
        Title = {SqueezeDet: Unified, Small, Low Power Fully Convolutional Neural Networks for Real-Time Object Detection for Autonomous Driving},
        Journal = {arXiv:1612.01051},
        Year = {2016}
    }


Adaptations have been made on the training algorithm and network structures to enable pruning. Also code has been added for pruning purposes and conversion to tflite for the Raspberry Pi.

## Installation:

- Clone the SqueezeDet repository:

  ```Shell
  git clone https://github.com/2ArounD/Pi-SqueezeDet.git
  ```


  - Use pip to install required Python packages:

    ```Shell
    pip install -r requirements.txt
    ```

## Data:

- This repository makes use of the KITTI 2d object detection dataset, this can be downloaded from http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d. Place the data in the folder ROOT/data/KITTI and unzip. Then run the data split script provided by the repository from Bichen. Create the necessary directories and run the python script:

```Shell
  cd $ROOT/data/KITTI/
  mkdir ImageSets
  cd ./ImageSets
  ls ../training/image_2/ | grep ".png" | sed s/.png// > trainval.txt
  ```
```Shell
  cd $ROOT/data/
  python random_split_train_val.py
  ```

This will result in the following folder and file structure with the KITTI data:

```Shell
  $SQDT_ROOT/data/KITTI/
                    |->training/
                    |     |-> image_2/00****.png
                    |     L-> label_2/00****.txt
                    |->testing/
                    |     L-> image_2/00****.png
                    L->ImageSets/
                          |-> trainval.txt
                          |-> train.txt
                          L-> val.txt
  ```

- The starting weights for training can be initialized with weights from the ImageNet SqueezeNet. These can be downloaded by running the following commmands:

```Shell
  cd $SQDT_ROOT/data/
  # SqueezeNet
  wget https://www.dropbox.com/s/fzvtkc42hu3xw47/SqueezeNet.tgz
  tar -xzvf SqueezeNet.tgz
  ```

## Training


## Pruning


## Testing on Raspberry Pi





