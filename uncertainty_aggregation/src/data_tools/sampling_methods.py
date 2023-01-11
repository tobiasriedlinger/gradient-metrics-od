
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import smogn
import pandas as pd

"""
While we implement the SMOGN updampling method form https://github.com/nickkunz/smogn
note that applying it in practice (especially in this project) takes an exceeding amount of time
and seems to not improve meta regression results by that much.
That is why in src.uncertainty_quantification.parameter_search.py we call None as
default_augmentation for regressors.
"""


def get_augmented_vars_and_targets(var_df, target_df, augmentation_method):
    if augmentation_method == "smote":
        oversample = SMOTE()
        return oversample.fit_resample(var_df, target_df)

    elif augmentation_method == "smote+rus":
        over = SMOTE(sampling_strategy=0.5)
        under = RandomUnderSampler(sampling_strategy=1.0)
        steps = [("o", over), ("u", under)]
        pipeline = Pipeline(steps=steps)
        return pipeline.fit_resample(var_df, target_df)

    elif augmentation_method == "smogn":
        smogn_df = smogn.smoter(data=pd.concat(
            [var_df, target_df], axis=1), y="true_iou")
        iou_df = smogn_df["true_iou"].copy(deep=True)
        variables_df = smogn_df.drop("true_iou", axis=1)

        return variables_df, iou_df
