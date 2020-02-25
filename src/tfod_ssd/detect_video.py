import argparse
import logging
import os
import time

import cv2
import imutils
from imutils.video import FPS
from imutils.video import VideoStream
import numpy as np

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

# the number of times we'll retry pulling a frame from the video
# source before giving up and assuming we've reached then end
_MAX_RETRIES = 10

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
        "--videosrc",
        type=str,
        required=False,
        default="0",
        help="RTSP URL for an IP camera, video (MP4) file path, or '0' for webcam",
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
    args_parser.add_argument(
        "--out_dir",
        type=str,
        required=False,
        help="path to directory where detection images should be written",
    )
    args = vars(args_parser.parse_args())

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

    # if we've specified to output images then we create the output directory,
    # initialize an output image count, and get a base name for the images
    if args["out_dir"] is not None:
        os.makedirs(args["out_dir"], exist_ok=True)
        detection_frame_count = 0
        if args["videosrc"] == "0":
            base_name = "webcam"
        else:
            base_name = os.path.splitext(os.path.split(args["videosrc"])[-1])[0]

    # initialize the video stream, wait a few seconds to allow
    # the camera sensor to warm up, and initialize the FPS counter
    _logger.info("starting video stream...")
    if args["videosrc"] == "0":
        video_source = 0
    else:
        video_source = args["videosrc"]
    video_stream = VideoStream(src=video_source).start()
    time.sleep(2.0)
    fps = FPS().start()
    cv2.waitKey(1) & 0xFF

    # get the initial frame, in order to have a baseline image for motion detection
    previous_frame = video_stream.read()
    if _RESIZE_WIDTH > 0:
        previous_frame = imutils.resize(previous_frame, width=_RESIZE_WIDTH)

    # flag for keeping a count of retries
    retries = 0

    # a list of detection (dictionary) records -- when this list
    # fills up to or above the batch size then we'll write the
    # entire list to our CSV file or add to the database
    detections_batch = []

    # loop over the frames from the video stream
    while True:

        # read the image frame and resize if necessary
        frame = video_stream.read()
        if frame is None:
            retries += 1
            if retries > _MAX_RETRIES:
                break
            else:
                continue
        else:
            retries = 0

        if _RESIZE_WIDTH > 0:
            frame = imutils.resize(frame, width=_RESIZE_WIDTH)

        # only perform detection if we've read a significantly different frame
        detections_made = False
        difference = np.sum(np.absolute(frame - previous_frame)) / np.size(frame)
        if difference > 50:

            # set up the previous frame for the next loop iteration
            previous_frame = frame

            # perform inference to get detections drawn on the frame
            frame, detections = inference(object_detector, frame, args["confidence"])

            # add the detections from this frame into the current batch
            if len(detections) > 0:
                detections_batch += detections
                detection_frame_count += 1
                detections_made = True

        # update the FPS counter
        fps.update()

        # we've now performed all detections (if any) on the frame

        # display the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # write the detection frame
        if detections_made and (args["out_dir"] is not None):
            detection_image_path = os.path.join(
                args["out_dir"], f"{base_name}_{detection_frame_count}.jpg",
            )
            cv2.imwrite(detection_image_path, frame)

        # if we've collected a batch of detections then write them to CSV
        if len(detections_batch) > _DETECTIONS_BATCH_SIZE:

            write_detections(detections_batch, args["csv"], detection_field_names)
            detections_batch = []

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # stop the timer and display FPS information
    fps.stop()
    _logger.info(f"elapsed time: {fps.elapsed():.2f}")
    _logger.info(f"approx. FPS: {fps.fps():.2f}")

    # do a bit of cleanup
    cv2.destroyAllWindows()
    video_stream.stop()


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Usage:
    # ~~~~~~~~~~~~~~~
    #
    # $ python <this_script.py> --graph <inference_graph.pb> \
    #     --labelmap label_map_prototext_path \
    #     [--videosrc camera_rtsp_url|mp4_file_path|0] \
    #     [--csv results_csv_file_path] \
    #     [--confidence threshold] \

    # launch the main function
    main()
