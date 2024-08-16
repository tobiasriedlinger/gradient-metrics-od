# Code used in Gradient-Based Quantification of Epistemic Uncertainty for Deep Object Detectors

We deliver all code used for our experimental setups and for the generation of the shown plots.
The folders yolov3-torch, faster-rcnn-torch and retinanet-torch contain the object detection pipelines.
We are unable to provide weight files for the networks due to upload constraints.
The object detectors all implement a pipeline for computing gradient uncertainty metrics, a more detailed structure can be found in the respective README.md-files included in the folders.

The folder uncertainty_aggregation contains the framework we used for aggregating uncertainty metrics in meta classification and meta regression, as well as the main experimental setups once gradient-based and Monte-Carlo dropout-based uncertainty metrics have been produced.
We included a more detailed description in the README.md-file in the folder itself.

Paper: [https://openaccess.thecvf.com/content/WACV2023/html/Riedlinger_Gradient-Based_Quantification_of_Epistemic_Uncertainty_for_Deep_Object_Detectors_WACV_2023_paper.html](https://openaccess.thecvf.com/content/WACV2023/html/Riedlinger_Gradient-Based_Quantification_of_Epistemic_Uncertainty_for_Deep_Object_Detectors_WACV_2023_paper.html)
