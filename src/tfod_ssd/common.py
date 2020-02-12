import csv
from typing import Dict, List

import arrow
import cv2
import numpy as np

from tfod_ssd.detector import ObjectDetectorTensorFlow

# width of bounding boxes (in pixels)
_BOX_LINE_WIDTH = 2

# color of bounding boxes
_BOX_COLOR = (0, 255, 0)

# width of labels (in pixels)
_LABEL_WIDTH = 2

# color of bounding boxes
_LABEL_COLOR = (255, 0, 0)


# ------------------------------------------------------------------------------
def _draw_boxes(
        frame: np.ndarray,
        detections: List[Dict],
) -> np.ndarray:
    """
    Draws bounding boxes corresponding to object detections onto an image frame.

    :param frame:
    :param detections:
    :return: array of pixels with bounding boxes and labels drawn
    """

    # for each detection draw a corresponding labeled bounding box
    for detection in detections:

        # draw the bounding box of the face
        frame = \
            cv2.rectangle(
                frame,
                (detection["start_x"], detection["start_y"]),
                (detection["end_x"], detection["end_y"]),
                _BOX_COLOR,
                _BOX_LINE_WIDTH,
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
                            _LABEL_COLOR,
                            _LABEL_WIDTH)

    return frame


# ------------------------------------------------------------------------------
def inference(
        detector: ObjectDetectorTensorFlow,
        frame: np.ndarray,
        confidence_threshold: float,
) -> (np.ndarray, List[Dict]):
    """
    Performs inference on an image frame using an object detector model,
    drawing bounding boxes with labels on the frame if any detections are
    made and returning a list of dictionaries specifying the detections.

    :param detector: object detection bject that will be used to perform inferencing
    :param frame: image frame of BGR pixel values
    :param confidence_threshold: threshold percentage required for a detection
    :return: the same image frame with bounding boxes drawn if any detections
        were made and a list of dictionary objects specifying the detections
    """

    # get the date/time
    time_stamp = arrow.utcnow().timestamp

    # list of detections for this frame
    detections_frame = []

    # get the dimensions
    (height, width) = frame.shape[:2]

    # loop over the object detections
    object_detections = detector.detect(frame, confidence_threshold)
    for object_detection in object_detections:

        # compute the detected object's bounding box (x, y)-coordinates
        # NOTE bounding boxes are in order (y0, x0, y1, x1)
        box = object_detection["bounding_box"] * np.array([height, width, height, width])
        (start_y, start_x, end_y, end_x) = box.astype("int")

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

    # draw bounding boxes on the frame to indicate detected objects
    if len(detections_frame) > 0:
        frame = _draw_boxes(frame, detections_frame)

    return frame, detections_frame


# ------------------------------------------------------------------------------
def write_detections(
        detections: List[Dict],
        detections_csv: str,
        detection_field_names: List[str],
):
    """
    Writes detections to a CSV file.

    :param detections:
    :param detections_csv:
    :param detection_field_names:
    """

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
