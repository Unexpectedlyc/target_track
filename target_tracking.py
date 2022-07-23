from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import random
import time
from utils.datasets import letterbox
import numpy as np
import torch
from numpy import random
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import time_synchronized,TracedModel
from PIL import Image, ImageDraw, ImageFont
import imutils
import cv2
import config

cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

#为每个类别生成一个color，存放在dict中
names=[ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donu t', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
color = [[random.randint(0, 255) for _ in range(3)] for _ in names]
color_dict=dict(zip(names,color))


#opencv添加中文字体
def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

#给检测结果可视化画矩形框
def plot_bboxes(image, bboxes,line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    num=0 #目标数量
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        class_name, _ = cls_id.rsplit(" ", 1)

        if pos_id>num:
            num=pos_id
        c1, c2 = (x1, y1), (x2, y2)
        #画矩形框
        cv2.rectangle(image, c1, c2, color_dict[class_name], thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        #添加标签名称置信度
        cv2.rectangle(image, c1, c2, color_dict[class_name], -1, cv2.LINE_AA)  # filled
        cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return image,num

#目标检测结果
def detect(img,classname=None):
    with torch.no_grad():
        device = config.device
        half = device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        model = attempt_load(config.model_path, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(config.image_size, s=stride)  # check img_size
        #if trace:
        #    model = TracedModel(model, device, 640)
        if half:
            model.half()  # to FP16
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()

        # Padded resize
        im0s=img
        img = letterbox(im0s, 640, stride=32)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, config.conf_thres, config.iou_thres, classes=None, agnostic=False)
        t2 = time_synchronized()

        # Process detections
        pred_boxes = []
        for i, det in enumerate(pred):  # detections per image

            im0=im0s

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):

                    # Add bbox to image
                    x1, y1 = int(xyxy[0]), int(xyxy[1])
                    x2, y2 = int(xyxy[2]), int(xyxy[3])
                    label = f'{names[int(cls)]} {conf:.2f}'

                    class_name, _ = label.rsplit(" ", 1)
                    if classname == None:
                        pred_boxes.append((x1, y1, x2, y2, label, conf))
                    elif class_name == classname:
                        pred_boxes.append((x1, y1, x2, y2, label, conf))


        print(f'one img cost time：({time.time() - t0:.3f}s)')
        return im0s,pred_boxes

#用deepsort追踪
def update_tracker(image,classname=None):
    _, bboxes= detect(image,classname)

    bbox_xywh = []
    confs = []
    clss = []
    bboxes2draw = []

    for x1, y1, x2, y2, cls_id, conf in bboxes:

        obj = [
            int((x1+x2)/2), int((y1+y2)/2),
            x2-x1, y2-y1
        ]
        bbox_xywh.append(obj)
        confs.append(conf)
        clss.append(cls_id)

    xywhs = torch.Tensor(bbox_xywh)
    confss = torch.Tensor(confs)

    outputs = deepsort.update(xywhs, confss, clss, image)

    for value in list(outputs):
        x1, y1, x2, y2, class_conf, track_id = value
        bboxes2draw.append((x1, y1, x2, y2, class_conf, track_id))

    image,num= plot_bboxes(image, bboxes2draw)

    return image,num

#返回鼠标点击位置坐标
def print_position(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)

def main(path,classname=None,IsvideoWriter = False):
    #classname不設置就是coco中全類別追蹤
    #IsvideoWriter默认False,设置为True则保存结果视频
    name = 'demo'
    number=0
    cap = cv2.VideoCapture(path)
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000/ fps)
    videoWriter=None
    #cv2.namedWindow(name)
    #点击返回坐标
    #cv2.setMouseCallback(name, print_position)

    while True:
        _, im = cap.read()
        if im is None:
            break
        result,num= update_tracker(im,classname)

        if num>number:
           number=num

        result= cv2AddChineseText(result, '目标数目:{}'.format(number), (50, 50), (0, 0, 255))
        result = imutils.resize(result, height=600)
        if IsvideoWriter:
            if videoWriter is None:
                fourcc = cv2.VideoWriter_fourcc(
                    'm', 'p', '4', 'v')  # opencv3.0
                videoWriter = cv2.VideoWriter(
                    'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))
            videoWriter.write(result)

        cv2.imshow(name, result)

        if cv2.waitKey(t) & 0xFF == 27:
            #64位操作系统按ESC停止
            break
    cap.release()
    if IsvideoWriter:
        videoWriter.release()
    cv2.destroyAllWindows()

