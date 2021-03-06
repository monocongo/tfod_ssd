import argparse
import logging
import os

import cv2
import imutils

from tfod_ssd.common import inference, write_detections
from tfod_ssd.detector import ObjectDetectorTensorFlow

# number of detections we'll batch together into a single write operation
_DETECTIONS_BATCH_SIZE = 20

# if positive then we'll resize video to this width
_RESIZE_WIDTH = -1

# width of bounding boxes (in pixels)
_BOX_LINE_WIDTH = 2

# color of bounding boxes
_BOX_COLOR = (0, 255, 0)

# ------------------------------------------------------------------------------
# set up a basic, global logger object which will write to the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d  %H:%M:%S",
)
_logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
def main():
    """
    Performs inference for object detection to a video stream (or file),
    saving detection information and associated imagery to a CSV file.

    """

    # parse the command line arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--graph",
        type=str,
        required=True,
        help="path to TensorFlow model inference graph protobuf file",
    )
    args_parser.add_argument(
        "--labelmap",
        type=str,
        required=True,
        help="path to TensorFlow label map",
    )
    args_parser.add_argument(
        "--images",
        type=str,
        required=True,
        nargs='+',
        help="path to one or more JPG images and/or directories "
             "containing JPG images",
    )
    args_parser.add_argument(
        "--confidence",
        type=float,
        required=False,
        default=0.6,
        help="Confidence threshold below which detections are not used",
    )
    args_parser.add_argument(
        "--csv",
        type=str,
        required=False,
        default="detections.csv",
        help="path to results CSV file",
    )
    args = vars(args_parser.parse_args())

    # get the image file paths
    image_file_paths = []
    for path in args["images"]:
        if os.path.isfile(path) and path.endswith(".jpg"):
            image_file_paths.append(path)
        elif os.path.isdir(path):
            file_names = os.listdir(path)
            for file_name in file_names:
                if file_name.endswith(".jpg"):
                    image_file_paths.append(os.path.join(path, file_name))

    # we'll write detection info to CSV using the following columns
    detection_field_names = [
        "time_stamp",
        "class",
        "confidence",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
    ]

    # if a GPU is available then setting this environment variable
    # makes it more likely that the keras-* models will use it
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # load the pre-trained object detection model
    _logger.info("\n\n\tLoading object detection model...\n")
    object_detector = ObjectDetectorTensorFlow(args["labelmap"], args["graph"])

    cv2.waitKey(1) & 0xFF

    # a list of detection (dictionary) records -- when this list
    # fills up to or above the batch size then we'll write the
    # entire list to our CSV file or add to the database
    detections_batch = []

    for image_file_path in image_file_paths:

        # read the image frame and resize if necessary
        frame = cv2.imread(image_file_path)
        if _RESIZE_WIDTH > 0:
            frame = imutils.resize(frame, width=_RESIZE_WIDTH)

        # perform inference to get detections drawn on the frame
        frame, detections = inference(object_detector, frame, args["confidence"])

        # add the detections from this frame into the current batch
        if len(detections) > 0:
            detections_batch += detections

        # we've now performed all detections (if any) on the frame

        # display the output frame
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)

    # do a bit of cleanup
    cv2.destroyAllWindows()

    # write the detections to CSV
    write_detections(detections_batch, args["csv"], detection_field_names)


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Usage:
    # ~~~~~~~~~~~~~~~
    #
    # $ python <this_script.py> --graph <inference_graph.pb> \
    #     --labelmap label_map_prototext_path \
    #     --images <image_file_paths_and_or_directories_containing_images> \
    #     [--csv results_csv_file_path] \
    #     [--confidence threshold] \

    # launch the main function
    main()
