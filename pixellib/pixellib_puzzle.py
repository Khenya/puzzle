# in this lesson we will use our pre trained model and detect out trained objects in live camera
# We copy some of the code from our previous lesson
import time
import pixellib
import cv2
import numpy as np
from pixellib.instance import custom_segmentation
from pixellib.custom_train import instance_custom_training
from pixellib.instance import instance_segmentation


# segment_image = custom_segmentation()
# segment_image.inferConfig(network_backbone="resnet101",num_classes=3,class_names=["BG", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"])
# segment_image.load_model("mask_rcnn_coco.h5") 

# train_maskRcnn = instance_custom_training()
# train_maskRcnn.modelConfig(network_backbone="resnet101",num_classes=3, batch_size=1)
# train_maskRcnn.load_pretrained_model("mask_rcnn_coco.h5") 

segmentation_model = instance_segmentation()
segmentation_model.load_model('mask_rcnn_coco.h5')

# lets load the camera

capture = cv2.VideoCapture(0)

segmentation_model.process_camera(capture, show_bboxes=True, show_frames=True, frames_per_second=15, frame_name="frame")

# while True:
#     ret , frame = capture.read() # read the frame

#     # analyse the frame using our model
#     segmask , out = segmentation_model.segmentFrame(frame, show_bboxes=True)

#     cv2.imshow("frame", frame)

#     if cv2.waitKey(25) & 0xff == ord('q'):
#         break

    # Añade un retraso de 0.1 segundos (100 milisegundos)
    # time.sleep(0.1)

# cv2.destroyAllWindows()   

