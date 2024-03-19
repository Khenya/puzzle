import pixellib
from pixellib.instance import instance_segmentation

segment_image = instance_segmentation()
segment_image.load_model("mask_rcnn_coco.h5")
segment_image.segmentImage("mano5.jpg", output_image_name = "mano_out5.jpg", show_bboxes = True)