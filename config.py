from utils.torch_utils import select_device
import torch

#可修改的参数
#deepsort的参数在deep_sort.yaml文件中修改

conf_thres=0.62
iou_thres=0.45
model_path="weights/yolov7.pt"
device=select_device('0' if torch.cuda.is_available() else 'cpu')
image_size=640 #必须为32的整数倍
