# tfod_ssd
Usage of the TensorFlow object detection API for training an SSD model using transfer learning with a custom dataset 

### Install the TensorFlow Object Detection API
1. Clone the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) 
from GitHub and set an environment variable to facilitate referencing the location:
    ```
    $ git clone git@github.com:tensorflow/models.git
    $ cd models
    $ export TFOD=`pwd`
    ```

2. The TensorFlow Object Detection API uses [protocol buffer](https://developers.google.com/protocol-buffers) 
files to configure model and training parameters. Before the framework can be used, 
the Protobuf libraries must be compiled: 
    ```
    $ sudo apt-get install protobuf-compiler
    $ cd $TFOD/research
    $ protoc object_detection/protos/*.proto --python_out=.
    ```

3. When running locally, the `research` and `research/slim` 
directories should be appended to the `PYTHONPATH` environment variable: 
    ```
    $ cd $TFOD/research
    $ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
    ```
4. In order to utilize COCO evaluation metrics to evaluate the accuracy of our model 
during training we'll need to install the COCO Python API into the TensorFlow API:
    ```
    $ git clone https://github.com/cocodataset/cocoapi.git
    $ cd cocoapi/PythonAPI
    $ sudo make
    $ cp -r pycocotools $TFOD/research/
    ```

### Create Python virtual environment
1. Create a Python virtual environment containing all necessary TensorFlow packages. 
In this example we'll use an environment created and managed using 
[Anaconda](https://www.anaconda.com/distribution/) but it's also possible to use 
a [standard Python virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
    ```
    $ conda create -n tfod_ssd python=3 --yes
    $ conda activate tfod_ssd
    ```
2. Install the TensorFlow package.

    For GPU:
    ```
    $ conda install tensorflow-gpu
    ```
    For CPU:
    ```
    $ conda install tensorflow
    ```

3. Install the remaining dependencies:
    ```
    $ conda install contextlib2 Cython jupyter lxml matplotlib pillow
    ```

### Create an experiment directory
Create a directory to contain the files used in this "experiment" (dataset, model 
configuration, class labels, etc.)
```
$ mkdir tfod_ssd
$ cd tfod_ssd
$ export EXPERIMENT=`pwd`
```

### Build a custom dataset

Acquire a dataset in TFRecord format. Datasets with annotation files in other formats 
can be converted/translated to TFRecord format using the [cvdata](https://github.com/monocongo/cvdata) 
Python package. We will create a directory in our experiment directory named `tfrecord`
for the TFRecord file(s).

As well as the TFRecord file(s) we should also have a labels file in protobuf text 
format that lists items containing class IDs and their corresponding labels. 
For example, if the TFRecord files for an animals dataset uses the IDs/labels (1: cat, 
2: dog, and 3: panda) then the labels file will look like so:
```
item {
 id: 1
 name: 'cat'
}
item {
 id: 2
 name: 'dog'
}
item {
 id: 3
 name: 'panda'
}
```

### Model configuration
The model we’ll in this example is the Single Shot Detector(SSD) with MobileNet 
model (optimized for inference on mobile) pretrained on the COCO dataset called 
[ssd_mobilenet_v2_quantized_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz) 
taken from the [TensorFlow Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

1. Download and unpack the model archive
    ```
    $ wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz
    $ tar -xzf ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz
    ```

2. Configure the model training configuration file by copying an example configuration 
from the TensorFlow OD API into another location (in our case a directory named 
`training`) and modify it for our training dataset.
    ```
    $ mkdir $EXPERIMENT/training
    $ cp $TFOD/research/object_detection/samples/configs/ssd_mobilenet_v2_quantized_300x300_coco.config $EXPERIMENT/training
    $ export CONFIG=$EXPERIMENT/training/ssd_mobilenet_v2_quantized_300x300_coco.config
    ```
   Update the file as follows:
   
   Line 9: set the number of classes to the number of labeled classes in our dataset
   ```
   num_classes: 5
   ```

   Line 156: set the path to the checkpoint file we'll use as the starting point for the training
   ```
   fine_tune_checkpoint: “/home/james/tfod_ssd/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/model.ckpt"
   ```

   Line 175: set the path to the training dataset TFRecord file(s)
   ```
   input_path: "/home/james/tfod_ssd/tfrecord/train.record-?????-of-00100
   ``` 

   Line 177 and 191: set the path to the dataset's label map protobuf text file
   ```
   label_map_path: "/home/james/tfod_ssd/tfrecord/label_map.pbtxt"
   ```
   
   Line 189: set the path to the evaluation dataset TFRecord file(s)
   ```
   input_path: "/home/james/tfod_ssd/tfrecord/eval.record-?????-of-00100
   ``` 
   
   Line 181: set the number of examples in the evaluation dataset
   ```
   num_examples: 8000
   ```
   
   In the `eval_config` section we will comment out the `max_evals` entry and add 
   the following line to enable the COCO evaluation metrics:
   ```
   metrics_set: "coco_detection_metrics"
   ```

### Train the model
```
$ cd $TFOD/research/object_detection
$ cp $TFOD/research/object_detection/legacy/train.py .
$ python train.py --train_dir=$EXPERIMENT/training --pipeline_config_path=$CONFIG
```
### TensorBoard
In order to monitor the training progress we can run TensorBoard. The various events 
used by TensorBoard are logged to `$EXPERIMENT/training`, hence we'll launch the 
program like so:
```bash
$ tensorboard --logdir=$EXPERIMENT/training
```
By default the port used is 6006, so we'll view the progress of the training on 
TensorBoard using http://localhost:6006 (replace `localhost` with the public IP 
address if running the training on a remote/cloud instance).

### Prepare for inference
Once we have a checkpoint file (after the model has completed training) then we 
can export/freeze the inference graph and then optimize for inference. For example, 
if the final model checkpoint occurs at 9085 steps, resulting in a collection of 
checkpoint files with prefix `$EXPERIMENT/training/model.ckpt-9085` (including 
files `model.ckpt-9085.data-00000-of-00001`, `model.ckpt-9085.index`, and 
`model.ckpt-9085.meta`) then the inference graph is frozen and optimized like so:
```bash
$ export EXPORT=$EXPERIMENT/training/export_9085
$ python $TFOD/research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path $CONFIG --trained_checkpoint_prefix $EXPERIMENT/training/model.ckpt-9085 --output_directory $EXPORT
$ python -m tensorflow.python.tools.optimize_for_inference --input=$EXPORT/frozen_inference_graph.pb --output=$EXPORT/optimized_graph.pb --input_names="input" --output_names="final_result"
```

### Perform inference
Utilize the CLI for inference on a single image, collection of images in a directory:
```bash
$ python src/tfod_ssd/detect_image.py --fig /home/james/experiments/tfod_ssd/training/export_20000/frozen_inference_graph.pb --labelmap /home/james/experiments/tfod_ssd/tfrecord_label_map.prototxt --images /home/james/data/test/imgs /home/james/data/test/rifle_00827.jpg
```
Utilize the CLI for inference on a video file, RTSP URL, or webcam:
```bash
$ python /home/james/git/tfod_ssd/src/tfod_ssd/detect_video.py --fig /home/james/experiments/tfod_ssd/training/export_20000/frozen_inference_graph.pb --labelmap /home/james/experiments/tfod_ssd/tfrecord_label_map.prototxt --videosrc example.mp4
```

### Optimization for Jetson Nano
In order to deploy on an NIVDIA Jetson Nano device we need to build and optimize 
a graph using [TensorRT](https://developer.nvidia.com/tensorrt).
1. Install the [NVIDIA TensorFlow/TensorRT](https://github.com/NVIDIA-AI-IOT/tf_trt_models) 
package:
    ```bash
   $ cd ${GIT_DIR}
   $ git clone --recursive https://github.com/NVIDIA-Jetson/tf_trt_models.git
   $ cd tf_trt_models
   $ ./install.sh python3
   ```   
