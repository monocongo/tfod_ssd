import csv
from typing import Dict, List

import cv2
import numpy as np

# width of bounding boxes (in pixels)
_BOX_LINE_WIDTH = 2

# color of bounding boxes
_BOX_COLOR = (0, 255, 0)

# width of labels (in pixels)
_LABEL_WIDTH = 2

# color of bounding boxes
_LABEL_COLOR = (255, 0, 0)


# ------------------------------------------------------------------------------
def draw_boxes(
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
