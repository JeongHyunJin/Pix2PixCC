# pix2pixCC model

The *pix2pixCC* is an improved deep learning model to use scientific datasets than previous models (*pix2pix* and *pix2pixHD*).
It uses update loss functions: those of *pix2pixHD* model and correlation coefficient (CC) values between the real and generated data.

The model consists of three major components: Generator, Discriminator, and Inspector.
The Generator and Discriminator are networks which get an update at every step with loss functions, and the Inspector is a module that guides the Generator to be well trained computing the CC values.
The Generator tries to generate realistic output from input, and the Discriminator tries to distinguish the more realistic pair between a real pair and a generated pair.
The real pair consists of real input and target data. The generated pair consists of real input data and output data from the Generator.

While the model is training, both networks compete with each other and get an update at every step with loss functions. 
Loss functions are objectives that score the quality of results by the model, and the networks automatically learn that they are appropriate for satisfying a goal, i.e., the generation of realistic data. 
They are iterated until the assigned iteration, which is a sufficient number assuring the convergence of the model.

<br/>

--------------

<br/>


## Environments

This code has been tested on Ubuntu 18.04 with one or two Nvidia GeForce GTX Titan XP GPU, CUDA Version 11.0, Python 3.6.9, and PyTorch 1.3.1.

<br/>


## Environments

* Linux or macOS
* Python 3
* NVIDIA GPU + CUDA cuDNN

<br/>

* Flags: see *pix2pixCC_Options.py* for all the training and test flags.
> Before running the model, you have to check or adjust the options for your input and target datasets.

     # data option in BaseOption class
     --dataset_name: 'pix2pixCC' (default)
     --data_format_input: 'tif' (default)
     --data_format_target: 'tif' (default)

     --input_ch: 1 (default)
     --target_ch: 1 (default)

     --data_size: 1024 (default)

     --logscale_input: False (default)
     --logscale_target: False (default)

     --saturation_lower_limit_input: -1 (default)
     --saturation_upper_limit_input: 1 (default)
     --saturation_lower_limit_target: -1 (default)
     --saturation_upper_limit_target: 1 (default)
      
>    And you have to set the pathes of input and target directories.

      # directory path for training in TrainOption class
      --input_dir_train: './datasets/Train/Input' (default)
      --target_dir_train: './datasets/Train/Target' (default)
      
>    &

      # directory path for test in TestOption class
      --input_dir_test: './datasets/Test/Input' (default)
      
      
      
<br/>

Getting Started
------------

**Installation**    
* Install Anaconada from https://docs.anaconda.com/anaconda/install/ (Our codes need numpy, scipy, astropy, and, random libraries)
* Install PyTorch and dependencies from http://pytorch.org
* Install Pillow with pip or conda ( https://pillow.readthedocs.io/en/stable/installation.html )


<br/>

**Dataset**       
* You need a large set of input & target image pairs for training.
* The input and target should be same pixel size.
* The size of height and width should be same. If the shape of your data is not square, you have to do cropping or padding the data before the training.
* The order of filenames is prepared to be in a sequence and should be same for the input and target data.

