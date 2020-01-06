# tfod_ssd
Usage of the TensorFlow object detection API for training an SSD model using transfer learning with a custom dataset 

### Install the TensorFlow Object Detection API
1. Clone the TensorFlow Object Detection API from GitHub and set an environment 
variable to facilitate referencing the location:
    ```
    $ git clone git@github.com:tensorflow/models.git
    $ cd models
    $ export TFOD=`pwd`
    ```

2. The Tensorflow Object Detection API uses Protobufs to configure model and training 
parameters. Before the framework can be used, the Protobuf libraries must be compiled: 
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
    $ conda install --user Cython
    $ conda install --user contextlib2
    $ conda install --user pillow
    $ conda install --user lxml
    $ conda install --user jupyter
    $ conda install --user matplotlib
    ```

### Build a custom dataset

Acquire a dataset in TFRecord format. Datasets with annotation files in other formats 
can be converted/translated to TFRecord format using the [cvdata](https://github.com/monocongo/cvdata) 
Python package. 

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

