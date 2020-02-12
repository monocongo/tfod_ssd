import argparse
import csv
import logging
import os
import time
from typing import Dict, List

import arrow
import cv2
import imutils
from imutils.video import FPS
from imutils.video import VideoStream
import numpy as np

from tfod_ssd.detector import ObjectDetectorTensorFlow

# number of detections we'll batch together into a single write operation
_DETECTIONS_BATCH_SIZE = 20

# ------------------------------------------------------------------------------
# set up a basic, global logger object which will write to the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d  %H:%M:%S",
)
_logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
def _draw_boxes(frame: np.ndarray,
                detections: List[Dict]) -> np.ndarray:

    line_width = 2

    # for each detection draw a corresponding labeled bounding box
    for detection in detections:

        # draw the bounding box of the face
        frame = \
            cv2.rectangle(
                frame,
                (detection["start_x"], detection["start_y"]),
                (detection["end_x"], detection["end_y"]),
                (0, 255, 0),
                2,
            )

        # draw the object's label and probability value
        label = f"{detection['class']}: {int(float(detection['confidence']) * 100)}%"
        if detection["start_y"] - 10 > 10:
            y = detection["start_y"] - 10
        else:
            y = detection["start_y"] + 10
        frame = cv2.putText(frame,
                            label,
                            (detection["start_x"], y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 255, 0),
                            2)

    return frame


# ------------------------------------------------------------------------------
def _write_detections(
        detections: List[Dict],
        detections_csv: str,
        detection_field_names: List[str],
):
    with open(detections_csv, "a") as detections_csv_file:
        detection_csv_writer = csv.DictWriter(
            detections_csv_file,
            fieldnames=detection_field_names,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
        )

        # write each detection record (dictionary) into a CSV file
        for detection in detections:
            detection_csv_writer.writerow(detection)
        detections_csv_file.flush()


# ------------------------------------------------------------------------------
def main():
    """
    Performs inference for object detection to a video stream (or file),
    saving detection information and associated imagery to a CSV file.

    """

    # parse the command line arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="path to TensorFlow model checkpoint",
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
    args = vars(args_parser.parse_args())

    # we're not storing detection records to database so we'll write detection info to CSV
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

    # load the pre-trained, serialized object detection model
    _logger.info("\n\n\tLoading object detection model...\n")

    object_detector = ObjectDetectorTensorFlow(args["labelmap"], args["checkpoint"])

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
    previous_frame = imutils.resize(previous_frame, width=400)

    # perform some casts so we don't need to do them at each iteration
    confidence_object = float(args["confidence"])

    # a list of detection (dictionary) records -- when this list
    # fills up to or above the batch size then we'll write the
    # entire list to our CSV file or add to the database
    detections_batch = []

    # loop over the frames from the video stream
    while True:

        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = video_stream.read()
        frame = imutils.resize(frame, width=400)

        # only perform detection if we've read a significantly different frame
        difference = np.sum(np.absolute(frame - previous_frame)) / np.size(frame)
        if difference > 50:

            # get the date/time
            time_stamp = arrow.utcnow().timestamp

            # set up the previous frame for the next loop iteration
            previous_frame = frame

            # list of detections for this frame
            detections_frame = []

            # get the dimensions
            (height, width) = frame.shape[:2]

            # loop over the object detections
            object_detections = object_detector.detect(frame, confidence_object)
            for object_detection in object_detections:

                # compute the detected object's bounding box (x, y)-coordinates
                box = object_detection["bounding_box"] * np.array([width, height, width, height])
                (start_x, start_y, end_x, end_y) = box.astype("int")

                # build a record for the object detection, add to the list
                detection = {
                    "time_stamp": time_stamp,
                    "class": object_detection["label"],
                    "confidence": f'{object_detection["probability"]:.2f}',
                    "start_x": start_x,
                    "start_y": start_y,
                    "end_x": end_x,
                    "end_y": end_y,
                }
                detections_frame.append(detection)

            # update the FPS counter
            fps.update()

            # draw bounding boxes on the frame to indicate detected objects
            if len(detections_frame) > 0:
                frame = _draw_boxes(frame, detections_frame)

                # add the detections from this frame into the current batch
                detections_batch += detections_frame

        # we've now performed all detections (if any) on the frame

        # display the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if we've collected a batch of detections then write them to CSV
        if len(detections_batch) > _DETECTIONS_BATCH_SIZE:

            _write_detections(detections_batch, args["csv"], detection_field_names)
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
    # $ python <this_script.py> --checkpoint model_ckpt_path \
    #     --labelmap label_map_prototext_path \
    #     [--videosrc camera_rtsp_url|mp4_file_path|0] \
    #     [--csv results_csv_file_path] \
    #     [--confidence threshold] \

    # launch the main function
    main()
