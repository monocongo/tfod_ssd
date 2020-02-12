from typing import Dict, List

import numpy as np
import tensorflow as tf


class ObjectDetectorTensorFlow:

    def __init__(
            self,
            labelmap: str,
            frozen_inference_graph: str,
    ):
        """
        Constructor function.

        :param labelmap: path to TFRecord labels map prototext file
        :param frozen_inference_graph: path to frozen inference graph protobuf file
        """

        # load the label map
        self.categories = self._parse_label_map(labelmap)

        # Load the TensorFlow model into memory
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(frozen_inference_graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.tf_session = tf.Session(graph=detection_graph)

        # the image will act as the input tensor
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # detection boxes, scores, number of objects
        # detected, and classes will be the output tensors
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    def detect(
            self,
            frame: np.ndarray,
            confidence: float,
    ) -> List[Dict]:
        """
        Get object detections from an image frame.

        :param numpy.ndarray frame: BGR image data array with shape
            (height, width, 3), with values in range (0..255), and dtype=uint8
        :param float confidence: minimum detection confidence (probability),
            used to filter weak detections
        :return: list of detection dictionaries, with each dictionary containing
            items "label", "probability", and "bounding_box"
        """

        # expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column
        # has the pixel BGR value
        frame_expanded = np.expand_dims(frame, axis=0)

        # perform object detection by running the model with the image as input
        (boxes, scores, classes, _) = \
            self.tf_session.run(
                [self.detection_boxes,
                 self.detection_scores,
                 self.detection_classes,
                 self.num_detections],
                feed_dict={self.image_tensor: frame_expanded},
            )

        # iterate over the detections, adding each to a list we'll return
        detections = []
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        for box, score, detection_class in zip(boxes, scores, classes):

            # if the probability score meets the confidence threshold
            # then add a detection to the list we'll return
            if score >= confidence:

                # create a dictionary for the detection and add to the list
                detection = {
                    "label": self.categories[int(detection_class)],
                    "probability": score,
                    "bounding_box": box,
                }
                detections.append(detection)

        return detections

    @staticmethod
    def _parse_label_map(
            labels_map: str,
    ) -> Dict:
        """
        Parses a labels map prototext file into a dictionary mapping class IDs
        to labels.

        :param labels_map: path to TFRecord labels map file
        :return: dictionary mapping class IDs to labels
        """

        categories = {}
        with open(labels_map) as label_map:

            for line in label_map:
                line = line.strip()
                if line.startswith("id"):
                    id_ = int(line.split(sep=":")[1])
                    name_line = label_map.readline().strip()
                    if name_line.startswith("name"):
                        categories[id_] = name_line.split(sep=":")[1].strip().strip('\'')
                    else:
                        raise ValueError("ID line not followed by name line")

        return categories
