import argparse
import json

import configs.uq_config as uq_conf
from api.loading_data import load_or_prepare_metrics
from src.uncertainty_quantification.train_and_evaluate import train_and_evaluate

parser = argparse.ArgumentParser(
    description="Evaluation for which parameter dictionary.")
parser.add_argument(dest="dict_path", type=str,
                    help="Path to parameter dictionary (.json file).")

if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.dict_path, "r") as f:
        parameter_dict = json.load(f)
    model = parameter_dict["model"]

    df, targets_df = load_or_prepare_metrics(
        model=model, metrics_const=parameter_dict["log"]["metrics"])
    print(df.columns)

    train_and_evaluate(df, targets_df, parameter_dict, uq_conf.df_path)
