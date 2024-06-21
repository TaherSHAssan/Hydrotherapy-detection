

!nvidia-smi

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.

# Download YOLOv7 repository and install requirements

# !git clone https://github.com/WongKinYiu/yolov7
# %cd yolov7
# !pip install -r requirements.txt

# current version of YOLOv7 is not compatible with pytorch>1.12.1 and numpy>1.20.1
# until the appropriate changes get made to the main repository, we will be using a fork containing the patched code
# you can track the progress here: https://github.com/roboflow/notebooks/issues/27
!git clone https://github.com/SkalskiP/yolov7.git
# %cd yolov7
!pip install -r requirements.txt

!pip install roboflow

from roboflow import Roboflow

# Commented out IPython magic to ensure Python compatibility.
# download COCO starting checkpoint
# %cd /content/yolov7
!wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt

# Commented out IPython magic to ensure Python compatibility.
# run this cell to begin training
# %cd /content/yolov7
!python train.py --batch 128 --epochs 50 --data /content/yolov7/Exercises-2/data.yaml --weights 'yolov7_training.pt' --device 0

from IPython.display import Image

display(Image("/content/yolov7/runs/train/exp/F1_curve.png", width=800, height=600))

display(Image("/content/yolov7/runs/train/exp/P_curve.png", width=800, height=600))

display(Image("/content/yolov7/runs/train/exp/PR_curve.png", width=800, height=600))

display(Image("/content/yolov7/runs/train/exp/confusion_matrix.png", width=800, height=600))

display(Image("/content/yolov7/runs/train/exp/R_curve.png", width=800, height=600))

!python detect.py --weights 'runs/train/exp/weights/best.pt' --conf 0.30 --img-size 640 --device '0' --source '/content/drive/MyDrive/20230404_131742A.mp4'

