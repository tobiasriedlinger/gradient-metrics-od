# Code used in Gradient-Based Quantification of Epistemic Uncertainty for Deep Object Detectors

We deliver all code used for our experimental setups and for the generation of the shown plots.
The folders yolov3-torch, faster-rcnn-torch and retinanet-torch contain the object detection pipelines.
We are unable to provide weight files for the networks due to upload constraints.
The object detectors all implement a pipeline for computing gradient uncertainty metrics, a more detailed structure can be found in the respective README.md-files included in the folders.

The folder uncertainty_aggregation contains the framework we used for aggregating uncertainty metrics in meta classification and meta regression, as well as the main experimental setups once gradient-based and Monte-Carlo dropout-based uncertainty metrics have been produced.
We included a more detailed description in the README.md-file in the folder itself.

ArXiv: [https://arxiv.org/abs/2107.04517](https://arxiv.org/abs/2107.04517)
