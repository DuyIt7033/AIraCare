# AIraCare
The website allows users to upload chest X-ray images and provides disease predictions using three pre-trained machine learning models. 

Window 10
python 3.10.8
tensorflow 2.10
tensorflow-gpu
cuda 11.2.0
cudnn 8.1.1.3

=======

Gpu = nvdia 1650 max-q (4g)
Ram 16g

Batch size = 16
learning rate 1e-4 to 1e-6
70 train - 20 val - 10 test
=========

Build  Resnet 50, vgg19 ,Densenet

rescare 1/255 [0;1]
['0_normal', '1_covid19', '2_Pneumonia']
Class Weights: {0: 0.9566850985396286, 1: 0.95587193653492, 2: 1.1006443193866733}

