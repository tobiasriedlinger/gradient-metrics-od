
import pandas as pd
import numpy as np
import tqdm
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.metrics import roc_auc_score, r2_score
import regression_tools as regtools


class MetaDetect(object):
    def __init__(self, metrics_frame, iou_frame):
        self.metrics_frame = metrics_frame
        self.iou = iou_frame
        if "Unnamed: 0" in self.metrics_frame.columns:
            self.metrics_frame = self.metrics_frame.drop("Unnamed: 0", axis=1)
        if "Unnamed: 0" in self.iou.columns:
            self.iou = self.iou.drop("Unnamed: 0", axis=1)
        self.standardize_metrics()

    def standardize_metrics(self):
        """
        Standardize uncertainty data such that mean(data) = 0 and std(data) = 1.
        """
        for col in self.metrics_frame.columns:
            dat = np.copy(np.array(self.metrics_frame[col]))
            self.metrics_frame[col] = (
                self.metrics_frame[col] - np.mean(dat)) / np.std(dat)

    def random_guessing_thresh(self, thresholds=[0.1, 0.3, 0.5, 0.7, 0.9], num_ensemble=10):
        """
        Performs a random guessing classification based on the true classification split that the ground truth has.
        :param thresholds: Iterable containing the running thresholds to train classifiers for.
        :param num_ensemble: Number of ensemble members to take mean/std acc/auroc over.
        :return:
        """
        cols = ["mean acc", "std acc", "mean auroc", "std auroc"]
        frame = pd.DataFrame(columns=cols, index=thresholds)

        iou_array = np.array(self.iou)

        for thresh in thresholds:
            targets = np.ravel(iou_array > thresh)

            accuracy = []
            auroc = []

            print("Aggregating random guessing classification for threshold {} over {} ensemble members...".format(
                thresh, num_ensemble))
            for i in tqdm.tqdm(range(num_ensemble)):
                guesses = np.random.rand(len(targets))
                prediction = (guesses <= thresh)
                accuracy.append(np.mean(prediction == targets))
                auroc.append(roc_auc_score(targets, prediction))

            metrics = [np.mean(accuracy), np.std(accuracy),
                       np.mean(auroc), np.std(auroc)]
            frame.loc[thresh, :] = metrics

        return frame

    def thresh_classification(self, thresholds=[0.1, 0.3, 0.5, 0.7, 0.9], method="logistic", num_ensemble=10, gb_depth=3, gb_n_estimators=100, gb_alpha=0.4, gb_lambda=0.4):
        assert method in ['logistic', 'gradient_boost']
        cols = ["mean val acc", "std val acc", "mean train acc", "std train acc",
                "mean val auroc", "std val auroc", "mean train auroc", "std train auroc"]
        frame = pd.DataFrame(columns=cols, index=thresholds)

        variables = np.copy(np.array(self.metrics_frame))
        iou_array = np.copy(np.array(self.iou))

        for thresh in thresholds:
            targets = np.ravel(iou_array > thresh)
            val_accuracy = []
            train_accuracy = []
            val_auroc = []
            train_auroc = []

            print("Aggregating {} classification for threshold {} over {} ensemble members...".format(
                method, thresh, num_ensemble))
            for i in tqdm.tqdm(range(num_ensemble)):

                val_mask = (np.random.rand(len(targets)) < 0.1)
                train_mask = np.logical_not(val_mask)
                variables_val, targets_val = variables[val_mask,
                                                       :], targets[val_mask]
                variables_train, targets_train = variables[train_mask,
                                                           :], targets[train_mask]

                if method == 'logistic':
                    model = LogisticRegression(
                        penalty='none', solver='saga', max_iter=3000, tol=1e-4).fit(variables_train, targets_train)
                elif method == 'gradient_boost':
                    model = XGBClassifier(verbosity=0, max_depth=gb_depth, colsample_bytree=0.5, n_estimators=gb_n_estimators,
                                          reg_alpha=gb_alpha, reg_lambda=gb_lambda).fit(variables_train, targets_train)

                predictions = model.predict(variables)
                prediction_probabilities = model.predict_proba(variables)
                val_accuracy.append(
                    np.mean(predictions[val_mask] == targets_val))
                train_accuracy.append(
                    np.mean(predictions[train_mask] == targets_train))
                val_auroc.append(roc_auc_score(
                    targets[val_mask], prediction_probabilities[val_mask, 1]))
                train_auroc.append(roc_auc_score(
                    targets[train_mask], prediction_probabilities[train_mask, 1]))

            metrics = [np.mean(val_accuracy), np.std(val_accuracy), np.mean(train_accuracy), np.std(
                train_accuracy), np.mean(val_auroc), np.std(val_auroc), np.mean(train_auroc), np.std(train_auroc)]
            frame.loc[thresh, :] = metrics

        return frame

    def regression(self, method="linear", num_ensemble=10, nn_epochs=30, gb_depth=3, gb_n_estimators=100, gb_alpha=0.4, gb_lambda=0.4):
        assert method in ['linear', 'shallow_nn', 'gradient_boost']

        r2_val = []
        r2_train = []

        variables = np.array(self.metrics_frame)
        iou_array = np.array(self.iou)

        num_variables = variables.shape[-1]
        iou_predictions = []

        print("Aggregating {} regression over {} ensemble members...".format(
            method, num_ensemble))
        for i in tqdm.tqdm(range(num_ensemble)):

            val_mask = np.random.rand(len(iou_array)) < 0.2
            train_mask = np.logical_not(val_mask)
            variables_val, variables_train = variables[val_mask,
                                                       :], variables[train_mask, :]
            targets_val, targets_train = iou_array[val_mask], iou_array[train_mask]

            if method == "linear":
                model = LinearRegression().fit(variables_train, targets_train)
            elif method == "shallow_nn":
                model = regtools.shallow_net_model(num_variables)
                model.compile(loss='mean_squared_error', optimizer='adam', metrics=[
                              regtools.shallow_net_stddev])
                model.fit(variables_train, targets_train,
                          epochs=nn_epochs, batch_size=128, verbose=0)
            elif method == "gradient_boost":
                model = XGBRegressor(verbosity=0, max_depth=gb_depth, colsample_bytree=0.5,
                                     n_estimators=gb_n_estimators, reg_alpha=gb_alpha, reg_lambda=gb_lambda).fit(
                    variables_train, targets_train)

            prediction = np.clip(model.predict(variables), 0, 1)
            if i == 0:
                iou_predictions.append(iou_array[val_mask])
                iou_predictions.append(prediction[val_mask])
            r2_val.append(r2_score(targets_val, prediction[val_mask]))
            r2_train.append(r2_score(targets_train, prediction[train_mask]))

        iou_predictions[0] = np.ravel(np.array(iou_predictions[0]))
        iou_predictions[1] = np.ravel(np.array(iou_predictions[1]))
        iou_predictions = pd.DataFrame(
            {"targets": iou_predictions[0], "predictions": iou_predictions[1]})
        r_squared_val, r_squared_train = np.array(r2_val), np.array(r2_train)
        frame = pd.DataFrame({"mean R^2 val": [np.mean(r_squared_val)],
                              "std R^2 val": [np.std(r_squared_val)],
                              "mean R^2 train": [np.mean(r_squared_train)],
                              "std R^2 train": [np.std(r_squared_train)]})

        return iou_predictions, frame

    def classification_best_gb_parameters(self, metric='mean val auroc', threshold=0.5, num_ensemble=10, depth_range=(2, 5), n_estim_range=(10, 25)):
        assert metric in ["mean val acc", "std val acc", "mean train acc", "std train acc",
                          "mean val auroc", "std val auroc", "mean train auroc", "std train auroc"]
        d_range = range(depth_range[0], depth_range[1]+1)
        n_range = range(n_estim_range[0], n_estim_range[1]+1)
        results = np.ndarray(shape=(len(d_range), len(n_range)))
        frames = []
        for d in range(len(d_range)):
            frames.append([])
            for n in range(len(n_range)):
                frames[d].append(self.thresh_classification(thresholds=[threshold], method='gradient_boost',
                                 num_ensemble=num_ensemble, gb_depth=d_range[d], gb_n_estimators=n_range[n]))
                results[d][n] = frames[d][n].loc[threshold][metric]
        d_max = np.argmax(results, axis=0)
        n_max = np.argmax(np.max(results, axis=0))
        d_max = d_max[n_max]

        return d_range[int(d_max)], n_range[int(n_max)]

    def regression_best_gb_parameters(self, metric='mean R^2 val', num_ensemble=5, depth_range=(2, 5), n_estim_range=(10, 25)):
        assert metric in ["mean R^2 val", "mean R^2 train"]
        d_range = range(depth_range[0], depth_range[1]+1)
        n_range = range(n_estim_range[0], n_estim_range[1]+1)
        results = np.ndarray(shape=(len(d_range), len(n_range)))
        frames = []
        for d in range(len(d_range)):
            frames.append([])
            for n in range(len(n_range)):
                frames[d].append(self.regression(
                    method="gradient_boost", num_ensemble=num_ensemble, gb_depth=d_range[d], gb_n_estimators=n_range[n]))
                results[d][n] = frames[d][n][1].loc[0][metric]
        d_max = np.argmax(results, axis=0)
        n_max = np.argmax(np.max(results, axis=0))
        d_max = d_max[n_max]

        return d_range[int(d_max)], n_range[int(n_max)]

    def quadratic_ridge(self, num_ensemble=10, alpha=1.0):
        r2_val = []
        r2_train = []

        variables = np.array(self.metrics_frame)
        iou_array = np.array(self.iou)

        iou_predictions = []

        print("Aggregating quadratic Ridge regression over {} ensemble members...".format(
            num_ensemble))
        for i in tqdm.tqdm(range(num_ensemble)):
            # np.random.seed(i)

            val_mask = np.random.rand(len(iou_array)) < 0.2
            train_mask = np.logical_not(val_mask)
            variables_val, variables_train = variables[val_mask,
                                                       :], variables[train_mask, :]
            targets_val, targets_train = iou_array[val_mask], iou_array[train_mask]

            model = Ridge(alpha=alpha).fit(variables_train, targets_train)

            prediction = np.clip(model.predict(variables), 0, 1)
            if i == 0:
                iou_predictions.append(iou_array[val_mask])
                iou_predictions.append(prediction[val_mask])
            r2_val.append(r2_score(targets_val, prediction[val_mask]))
            r2_train.append(r2_score(targets_train, prediction[train_mask]))

        iou_predictions[0] = np.ravel(np.array(iou_predictions[0]))
        iou_predictions[1] = np.ravel(np.array(iou_predictions[1]))
        iou_predictions = pd.DataFrame(
            {"targets": iou_predictions[0], "predictions": iou_predictions[1]})
        r_squared_val, r_squared_train = np.array(r2_val), np.array(r2_train)
        frame = pd.DataFrame({"mean R^2 val": [np.mean(r_squared_val)],
                              "std R^2 val": [np.std(r_squared_val)],
                              "mean R^2 train": [np.mean(r_squared_train)],
                              "std R^2 train": [np.std(r_squared_train)]})

        return iou_predictions, frame

    def thresh_gb_hierarchy(self, threshold=0.5, gb_depth=3, gb_n_estimators=100, gb_alpha=0.4, gb_lambda=0.4):
        variables = np.copy(np.array(self.metrics_frame))
        iou_array = np.copy(np.array(self.iou))

        targets = np.ravel(iou_array > threshold)

        print('Training gradient boosting for epsilon = {}, d = {}, n = {}...'.format(
            threshold, gb_depth, gb_n_estimators))
        val_mask = (np.random.rand(len(targets)) < 0.2)
        train_mask = np.logical_not(val_mask)
        variables_val, targets_val = variables[val_mask, :], targets[val_mask]
        variables_train, targets_train = variables[train_mask,
                                                   :], targets[train_mask]

        model = XGBClassifier(verbostiy=0, max_depth=gb_depth, colsample_bytree=0.5, n_estimators=gb_n_estimators,
                              reg_alpha=gb_alpha, reg_lambda=gb_lambda).fit(variables_train, targets_train)

        predictions = model.predict(variables)
        prediction_probabilities = model.predict_proba(variables)
        val_accuracy = np.mean(predictions[val_mask] == targets_val)
        train_accuracy = np.mean(predictions[train_mask] == targets_train)
        val_auroc = roc_auc_score(
            targets_val, prediction_probabilities[val_mask, 1])
        train_auroc = roc_auc_score(
            targets_train, prediction_probabilities[train_mask, 1])

        return model, val_accuracy, train_accuracy, val_auroc, train_auroc

    def train_classifier(self, threshold=0.5, gb_depth=3, gb_n_estimators=15, gb_alpha=0.4, gb_lambda=0.4):
        variables = np.copy(np.array(self.metrics_frame))
        iou_array = np.copy(np.array(self.iou))

        targets = np.ravel(iou_array > threshold)
        model = XGBClassifier(verbosity=0, max_depth=gb_depth, colsample_bytree=0.5, n_estimators=gb_n_estimators,
                              reg_alpha=gb_alpha, reg_lambda=gb_lambda).fit(variables, targets)

        return model
