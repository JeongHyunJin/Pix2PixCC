# *pix2pixCC*

The *pix2pixCC* is an improved deep learning model to use scientific datasets than previous approaches (*pix2pix* and *pix2pixHD*).
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

     # data setting in BaseOption class
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
> 

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


<br/>

**Training**   
* When the model is training, it saves the model every step (a step: saving frequency) as a file with an extension .pt or .pth at "./checkpoints/*dataset_name*/Model"
* You can set the saving frequency in *pix2pixCC_Options.py*. If you define "save_freq" of 10000, for example, a file which have an extension .pt will be saved every 10000 iterations.
* It will save a pair of images for the Real data and Generated one by the model every specified step at "./checkpoints/*dataset_name*/Image/Train". You can define the steps from "display_freq" in *pix2pixCC_Options.py*.
* The number of epoch is a hyperparameter that defines the number times that the deep learning model works through the entire training dataset. (the number of epoch) = (iterations) / (the number of dataset).
* "n_epochs" in *pix2pixCC_Options.py* should be larger than ("save_freq") / (The number of training input datasets).


<br/>

You can train the model with manually modified options as below:

Ex 1)
   
    python pix2pixCC_Train.py \
    --dataset_name 'EUV2Mag' \
    --data_format_input 'fits' \
    --data_format_target 'fits' \
    --data_size 1024 \
    --input_ch 3 \
    --logscale_input True \
    --saturation_lower_limit_input 1 \
    --saturation_upper_limit_input 200 \
    --saturation_lower_limit_target -3000 \
    --saturation_upper_limit_target 3000 \
    --input_dir_train '../Datasets/Train_data/Train_input' \
    --target_dir_train '../Datasets/Train_data/Train_output' \
    --n_epochs 100
    
<br/>
   
Ex 2)
   
    python pix2pixCC_Train.py \
    --dataset_name 'Map2Sim' \
    --data_size 256 \
    --input_dir_train 'D:/Train_input' \
    --target_dir_train 'D:/Train_output' \
    --norm_type 'BatchNorm2d' \
    --batch_size 64 \
    --save_freq 100 \
    --n_epochs 100
    
<br/>

**Test**     
* It will save the AI-generated data every step (a step: saving frequency) at "./checkpoints/*dataset_name*/Image/Test"
* When you set an iteration in TestOption class of *pix2pixHD_Options.py*, it saves the generated data by a model which saved before.
* BaseOptions in *pix2pixCC_Options.py* when you train the model and when you test the model should be same.

Ex 1)

     python pix2pixCC_Test.py \
     --dataset_name 'EUV2Mag' \
     --data_format_input 'fits' \
     --data_format_target 'fits' \
     --data_size 1024 \
     --input_ch 3 \
     --logscale_input True \
     --saturation_lower_limit_input 1 \
     --saturation_upper_limit_input 200 \
     --saturation_lower_limit_target -3000 \
     --saturation_upper_limit_target 3000 \
     --input_dir_test '../Datasets/Test_data/Test_input' \
     --iteration 100000
    
<br/>

Ex 2)

    python pix2pixCC_Test.py \
    --dataset_name 'Map2Sim' \
    --data_size 256 \
    --input_dir_test 'D:/Test_input' \
    --norm_type 'BatchNorm2d' \
    --batch_size 64 \
    --save_freq 100 \
    --n_epochs 100
<br/>


**Outputs**   
   It will make directories and save outputs as below:
    
    # pix2pixCC_Train.py:
       ./chechpoints/{dataset_name}/Image/Train/{iteration_real.png}
       ./chechpoints/{dataset_name}/Image/Train/{iteration_fake.png}
       ./chechpoints/{dataset_name}/Model/{iteration_G.pt}
       ./chechpoints/{dataset_name}/Model/{iteration_D.pt}
>

    # pix2pixCC_Test.py:
       ./chechpoints/{dataset_name}/Image/Test/{iteration}/{input_filename_AI.extension}

<br/>

-------------------------

<br/>


Network architectures and Hyperparameters
------------

You can run this code by changing the hyperparameters of *pix2pixCC*.

<br/>

**Generator** 

      # network setting in BaseOption class
     --n_downsample: 4 (default)
     --n_residual: 9 (default)
     --trans_conv: True (default)

<br/>

**Discriminator** 

     # network setting in BaseOption class
     --n_D: 1 (default)
     

**Inspector** 

     # network setting in BaseOption class
     --n_CC: 2 (default)
     
<br/>

When the GPU memory is not enough, you can try reducing the number of channels in the first layer of networks. (e.g. --n_gf 32 --n_df 32)
   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The number of channels in the first layer of the Generator: n_gf <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The number of channels in the first layer of the Discriminator: n_df <br/>

     # network option in BaseOption class
     --n_gf: 64 (default)
     --n_df: 64 (default)
     
<br/>
<br/>

**Hyperparameters** 

* The loss configuration of the objective functions 


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Total loss = *lambda_LSGAN* * (LSGAN loss) + *lambda_FM* * (Feature Matching loss) + *lambda_CC* * (1 - CC value)   <br/>

      # hyperparameters in TrainOption class
      --lambda_LSGAN: 2.0 (default)
      --lambda_FM: 10.0 (default)
      --lambda_CC: 5.0 (default)

<br/>

* Optimizer    

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Optimizer : Adam solver <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; momentum beta 1 parameter : beta1 <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; momentum beta 2 parameter : beta2 <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Learning rate : lr <br/>


      # hyperparameters in TrainOption class
      --beta1: 0.5 (default)
      --beta2: 0.999 (default)
      --lr: 0.0002 (default)

<br/>

* Initializer

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Initialize Weights in Convolutional Layers : normal distribution, mean : 0.0, standard deviation : 0.02   

<br/>
