import argparse
import logging
import os

from tf_trt_models.detection import download_detection_model
from tf_trt_models.detection import build_detection_graph
from tensorflow.python.util import deprecation
import tensorflow.contrib.tensorrt as trt
import tensorflow as tf

_BATCH_SIZE = 1

# ------------------------------------------------------------------------------
# set up a basic, global logger object which will write to the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d  %H:%M:%S",
)
_logger = logging.getLogger(__name__)


# turn off the deprecation warnings and logs to keep
# the console clean for convenience
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# ------------------------------------------------------------------------------
def main():
    # construct the argument parser and parse the arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to model configuration file",
    )
    args_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="path to model checkpoint (base path for .index, .meta, .data, etc.)",
    )
    args_parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="minimum detection probability to filter weak detections",
    )
    args_parser.add_argument(
        "--trt_graph",
        type=str,
        required=True,
        help="path where the TensorRT graph will be saved",
    )
    args = vars(args_parser.parse_args())

    # prepare the TensorFlow graph from the model configuration and checkpoint
    _logger.info("Preparing the TensorFlow graph...")
    frozen_graph, input_names, output_names = build_detection_graph(
        config=args["config"],
        checkpoint=args["checkpoint_path"],
        score_threshold = args["confidence"],
        batch_size = _BATCH_SIZE,
    )

    # create the optimized TRT graph from the frozen TensorFlow graph
    _logger.info("Creating TRT graph...")
    trt_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph,
        outputs=output_names,
        max_batch_size=_BATCH_SIZE,
        max_workspace_size_bytes=1 << 25,
        precision_mode="FP16",
        minimum_segment_size=50,
    )

    # serialize the TRT graph
    _logger.info("Serializing the TensorRT graph...")
    with open(args["trt_graph"], "wb") as f:
        f.write(trt_graph.SerializeToString())

    _logger.info(f"Model graph optimized for TensorRT is complete: {args['trt_graph']}")


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # USAGE
    # python optimize.py --config $EXPERIMENT/config/ssd_mobilenet_v2_quantized_300x300_coco.config \
    #     --checkpoint $EXPERIMENT/training/model.ckpt-45765 \
    #     --confidence 0.5 \
    #     --trt_graph $EXPERIMENT/trt/ssd_mobilenet_v2_quantized_300x300_coco_trt.pb

    main()
