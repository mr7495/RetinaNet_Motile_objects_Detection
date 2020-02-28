# RetinaNet_Detection
Implementation of the multi-frames detection method of motile objects

This repository is based on the https://github.com/fizyr/keras-retinanet and has been improved for motile objects detection.

The code has been tested on video samples with 25 frames and the sample of used annotations has been shared in annotation sample.csv file.
If you have more than 25 frames or want to use different type of annotation file change load-image function in keras_retinanet/preprocessing/csv_generator.py address.
In this code 3 consecutive frames have been concatenated to be used as the input of RetinaNet.
If you want to use more than 3 consecutive frames you have to apply some changes in these files:
1-resnet_retinanet function in keras_retinanet/models/resnet.py (change first layer shape)
2-load_image function in keras_retinanet/preprocessing/csv_generator.py (change the code to load more than 3 consecutive frames.)
3-keras_retinanet/utils/eval.py ( This line: image1 = generator.load_image(i)[:,:,1].astype(np.uint8),1 is the middle frame when we have 3 consecutive frames.)
4-preprocess_image function in keras_retinanet/utils/image.py

The idea is from paper https://arxiv.org/abs/2002.04034.

Cite as 	arXiv:2002.04034

Developer Email: mr7495@yahoo.com
