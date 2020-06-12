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

- Clone the Pi-SqueezeDet repository:

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

- Training can be started with the following command:

    ```Shell
    python3 ./src/train.py \
      --pretrained_model_path=[path/to/file/with/weights] \
      --data_path=[path/to/data/KITTI]\
      --image_set=train \
      --train_dir=[path/to/folder/where/training/output/is/stored] \
      --max_steps=2000 \
      --net=[squeezeDet+| squeezeDet+PruneFilterShape| squeezeDet+PruneFilter| squeezeDet+PruneLayer] \
      --summary_step=100 \
      --checkpoint_step=200 \
      --checkpoint_dir=[optional/path/to/folder/containing/ckpt/file/to/continue] \
      --pruning=False
    ```

    This command will start a training of the specified network on the KITTI dataset. For the initial training the ImageNet weights can be used in combination with the original network net=squeezeDet+. For training on pruned networks(see next section) the pickle files of pruned network weights can be used with the corresponding network.

## Evaluation

- For evaluation of the trained networks GPU_eval.py can be run with the following command:

    ```Shell
    python3 ./src/GPU_eval.py \
      --image_set=val\
      --eval_dir=[path/to/folder/where/eval/output/will/be/stored] \
      --pretrained_model_path=[path/to/pickle/file/with/weights] \
      --data_path=[path/to/data/KITTI]\
      --net=[squeezeDet+| squeezeDet+PruneFilterShape| squeezeDet+PruneFilter| squeezeDet+PruneLayer] \
      --checkpoint_path=[/path/to/folder/containing/ckpt/]
    ```

    This will output the boxes and classes of the validation set.

- By running the script run_kitti_cpp.py the offication kitti evaluation script can be started. This script needs the boxes and classes calculated by GPU_eval.py.

    ```Shell

      python3 ./src/run_kitti_cpp.py \
      --eval_dir=[path/to/folder/where/eval/output/is/stored] \
      --pretrained_model_path=[path/to/pickle/file/with/weights] \
      --data_path=[path/to/data/KITTI]\
    ```


## Pruning

- To prune a (trained) network the training script is used again with IS_PRUNING set to true. The script is started by the following command:

    ```Shell

    python3 ./src/train.py \
      --pretrained_model_path=[path/to/file/with/weights] \
      --data_path=[path/to/data/KITTI]\
      --image_set=train \
      --train_dir=[path/to/folder/where/pruning/output/is/stored] \
      --max_steps=2000 \
      --net=[squeezeDet+PruneFilterShape| squeezeDet+PruneFilter| squeezeDet+PruneLayer] \
      --summary_step=100 \
      --checkpoint_step=200 \
      --checkpoint_dir=[/path/to/folder/containing/ckpt/file/from/training] \
      --pruning=True
    ```

    The pruning script traines the network with a regulizer applied to the chosen structure. The to regularize structure can be chosen with the --net parameter. The networks weights are not updated during pruning, only the regularization paramters connected to the structures.

- After running the pruning script a new weights file can be generated with one of the following scripts, according to the pruned structure:

    - remove_filters.py
    - remove_layers.py
    - remove_rows_and_columns.py

    Running the scripts with the appropriate paths(in script), will result in a pickled dictionary with the new network structure and weights. This dictionary can be used as --pretrained_model_path in the python scripts to create a new and smaller graph in Tensorflow. When a pruned model is used, the appropriate --net should be selected as well. This will lead to smaller graphs with less parameters and actual speed-ups on your device.


## Testing on Raspberry Pi

To test the models on a Raspberry Pi, scripts to convert a trained and pruned model to the tflite format are included as well.

- First run the script lite-saved-model.py with the appropiate paths in the script. This will create a tensorflow SavedModel of the network with the right inputs and outputs.

- Then run the script lite-tflite-model.py with the appropiate paths in the script. This will convert the tensorflow SavedModel to a tflite model.

To use a tflite model on a Raspberry Pi, the Raspberry Pi had Raspian Buster Lite as OS and had the following packages installed:

- pillow 6.1.0
- numpy 1.17.3
- tflite_runtime 1.14.0

This will be enough to run the tflite model on the Raspberry Pi.


## Acceleration on GPU vs Raspberry Pi

Below the acceleration results after 3 iterations of pruning and retraining are shown. Removing complete filters works better relative to a gpu. Also absolute this is the best acceleration method for network pruning for a Raspberry Pi. Combining methods or increasing the pruning and retraining iterations could decrease the inference time even more.

![alt text](https://github.com/2ArounD/Pi-SqueezeDet/blob/master/results_3_iterations.png)


