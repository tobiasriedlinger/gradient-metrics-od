"""
Config file from old implementation. Redundant.
"""

# The configuration of COCO dataset
MISSING_IDS = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]
NUM_CLASSES_COCO = 80

# the parameters below are used to configure the YOLOv3 network correctly.
# Original YOLOv3 network has three scales, where each scale has three predefined anchors.
# As a result, nine anchors are needed.
# As a convention, the anchors are sorted from small to large.
# In the coco dataset, there are in total 80 classes.
# number of attributes are therefore 85, including four for bounding boxes and one for confidence.
SCALES = 3
NUM_ANCHORS_PER_SCALE = 3
ANCHORS = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]
assert len(ANCHORS) == SCALES * NUM_ANCHORS_PER_SCALE
NUM_CLASSES = NUM_CLASSES_COCO
NUM_ATTRIB = 4 + 1 + NUM_CLASSES
LAST_LAYER_DIM = NUM_ANCHORS_PER_SCALE * NUM_ATTRIB


# Training parameters.
# IGNORE_THRESH is the threshold whether to consider a certain detection is considered as non-object.
# As described in YOLOv3, "If the bounding box prior is not the best but does overlap a ground truth object
# by more than some threshold, we ignore the prediction." What the author really means is that:
# If the max IOU between the raw detection and all the ground truths is larger than IGNORE_THRESH, but not the best
# IOU among all the candidate detections with this ground truth, then we consider this detection will not contribute
# to the loss function.
IGNORE_THRESH = 0.5
# NOOBJ_COEFF and COORD_COEFF are the hyperparameters for the loss function, as described in YOLOv1 paper.
# Here we used the follow two values, which give comparable results with the original YOLOv3 implementation.
NOOBJ_COEFF = 0.2
COORD_COEFF = 5


# EPSILON is used to avoid computation instability like NaN or Inf.
EPSILON = 1e-9
