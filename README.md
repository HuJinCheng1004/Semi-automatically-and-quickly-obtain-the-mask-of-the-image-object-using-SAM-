# Semi-automatically-and-quickly-obtain-the-mask-of-the-image-object-using-SAM-
Using SAM image segmentation technology, manually click on the object position to mark it, and finally output the black and white mask photo of the object in png format

Note: python>=3.8, pytorch>=1.7, torchvision>=0.8

Feel free to ask any question. If you encounter any problem, please feel free to discuss in the comment area.

Official tutorial:
https://github.com/facebookresearch/segment-anything

1 Environment Configuration

1.1 Install the main libraries:
(1) pip:

Errors may occur, and Git needs to be configured properly.

pip install git+https://github.com/facebookresearch/segment-anything.git

(2) Local installation:

Errors may occur and Git needs to be configured properly.

git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .

(3) Manual download + manual local installation:
![image](https://github.com/user-attachments/assets/4de9dcaa-c925-4323-a69b-39914dcd8ea2)

zip file:

Link: https://pan.baidu.com/s/1dQ--kTTJab5eloKm6nMYrg
Extraction code: 1234
After decompression, run:
cd segment-anything-main
pip install -e .

1.2 Install dependent libraries:
pip install opencv-python pycocotools matplotlib onnxruntime onnx
matplotlib 3.7.1 and 3.7.0 may report an error

If an error occurs: pip install matplotlib==3.6.2

1.3 Download weight file:
Download one of the three weight files, I used the first one.

default or vit_h: ViT-H SAM model.
vit_l: ViT-L SAM model.
vit_b: ViT-B SAM model.
If the download is too slow:

Link: https://pan.baidu.com/s/11wZUcjYWNL6kxOH5MFGB-g
Extraction code: 1234


2 Tutorial

2.1 Extract objects based on the points selected on the image
Original image:
![image](https://github.com/user-attachments/assets/644bd9cc-4cd0-4112-b990-4e34388f9da2)

Read the image and select the cutout point:
(You can select one point or multiple points)
![image](https://github.com/user-attachments/assets/d5c2d848-d758-4ab8-af8a-364358345625)

Generate images and scores：
![image](https://github.com/user-attachments/assets/5f49d7d3-5134-4ecb-92bd-b9928253276b)

Save to local：
![mask_2_score_1 008](https://github.com/user-attachments/assets/59ba0887-fc75-4784-8ccd-ab389cf3d974)



