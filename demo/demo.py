# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from adet.config import get_cfg

from argparse import Namespace

# constants
WINDOW_NAME = "COCO detections"
import pickle

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    print(cfg.DATASETS)
    cfg.merge_from_file(args.config_file)
    #cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.WEIGHTS = args.model
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.DATASETS.TEST = ("kitchen_dataset_test",)
    cfg.MODEL.SOLOV2.NUM_CLASSES = 3
    cfg.freeze()
    print(cfg.DATASETS)
    return cfg


# def get_parser():
#     parser = argparse.ArgumentParser(description="Detectron2 Demo")
#     parser.add_argument(
#         "--config-file",
#         default="../configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
#         metavar="FILE",
#         help="path to config file",
#     )
#     parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
#     parser.add_argument("--video-input", help="Path to video file.")
#     parser.add_argument("--input", nargs="+", help="A list of space separated input images")
#     parser.add_argument(
#         "--output",
#         help="A file or directory to save output visualizations. "
#         "If not given, will show output in an OpenCV window.",
#     )
#
#     parser.add_argument(
#         "--confidence-threshold",
#         type=float,
#         default=0.3,
#         help="Minimum score for instance predictions to be shown",
#     )
#     parser.add_argument(
#         "--opts",
#         help="Modify config options using the command-line 'KEY VALUE' pairs",
#         default=[],
#         nargs=argparse.REMAINDER,
#     )
#     return parser

from detectron2.data.datasets import register_coco_instances

if __name__ == "__main__":

    #Fully overlapping objects
    image_path = "/home/enigma/Downloads/test_images/fully_stacked/IMG20210611194653.jpg"

    #full sink images
    #image_path = "/home/enigma/Downloads/test_images/full_sink_image/2021-07-14_14-54-57.jpg"

    #Mugs
    #image_path = "/home/enigma/Downloads/test_images/mug/coffeemug240.jpg"

    #image_path = "/home/enigma/Downloads/test_images/fully_stacked/IMG_1226.jpg" #False positive

    #Plates
    #image_path = "/home/enigma/Downloads/test_images/plates/plate831.jpg"
    image_path = "/home/enigma/Downloads/test_images/IMG20210529112600.jpg"
    #image_path = "/home/enigma/Downloads/test_images/IMG20210529112657.jpg"

    #Bowls
    #image_path = "/home/enigma/Downloads/test_images/bowls/Bowl54.jpg"
    #image_path = "/home/enigma/Downloads/test_images/bowls/img_1830.jpg"
    #image_path = "/home/enigma/Downloads/test_images/bowls/img_30.jpg" #plate in arm

    #Partially overlapped
    #image_path = "/home/enigma/Downloads/test_images/fully_stacked/IMG20210611194531.jpg"

    #image_path = "/home/enigma/Downloads/fov.png"

    #Synthetic datasets
    #image_path = "/home/enigma/Downloads/Perception/unity_labels/labels/Instance_2_1.jpg"

    image_path = "/home/enigma/Downloads/test_images/plates/plate854.jpg" #Single plate

    args = Namespace(config_file="/home/enigma/projects/asink/AdelaiDet/configs/SOLOv2/R50_3x.yaml",
                                   input=image_path,output=False,
                                               confidence_threshold = 0.1,
                                                                        model = "/home/enigma/Downloads/batch_4_2_model_0014999.pth")

    # register_coco_instances("kitchen_dataset_train", {},
    #                         "/content/AdelaiDet/datasets/kitchen_dataset_01_08_21_train_val/train_coco/annotations.json",
    #                         "/content/AdelaiDet/datasets/kitchen_dataset_01_08_21_train_val/train_coco/")
    # register_coco_instances("kitchen_dataset_val", {},
    #                         "/content/AdelaiDet/datasets/kitchen_dataset_01_08_21_train_val/val_coco/annotations.json",
    #                         "/content/AdelaiDet/datasets/kitchen_dataset_01_08_21_train_val/val_coco/")

    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    # Save the pickle file to be used in SOLOv2 module.
    with open('cfg_blender_class_2.pkl', 'wb') as outp:
        pickle.dump(cfg, outp, pickle.HIGHEST_PROTOCOL)
    # cfg = None
    # with open('cfg.pkl', 'rb') as inp:
    #     cfg = pickle.load(inp)

    print("Classes: {}".format(cfg.MODEL.ROI_HEADS.NUM_CLASSES))
    demo = VisualizationDemo(cfg, confidence_threshold=0.5)

    if args.input:
        # if os.path.isdir(args.input[0]):
        #     args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
        # elif len(args.input) == 1:
        #     args.input = glob.glob(os.path.expanduser(args.input[0]))
        #     assert args.input, "The input path(s) was not found"

        # use PIL, to be consistent with evaluation
        image_dir = "/home/enigma/Downloads/test_images/plates/"
        for image_name in os.listdir(image_dir):
            img_path = os.path.join(image_dir, image_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            start_time = time.time()
            h, w = img.shape[:2]
            neww = min(900, w)
            newh = int(neww * (h / w))
            #img = cv2.resize(img, (neww, newh))
            #resized_image = cv2.resize(img, (neww, newh))

            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: detected {} instances in {:.2f}s".format(
                    img_path, len(predictions["instances"]), time.time() - start_time
                )
            )
            output_image = visualized_output.get_image()[:, :, ::-1]
            h, w = output_image.shape[:2]
            neww = min(900, w)
            newh = int(neww * (h / w))
            img = cv2.resize(img, (neww, newh))
            output_image = cv2.resize(output_image, (neww, newh))
            cv2.imshow(WINDOW_NAME, output_image)
            cv2.waitKey(0)