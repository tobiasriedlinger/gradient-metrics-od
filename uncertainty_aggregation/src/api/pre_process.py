import numpy as np

from configs.uq_config import STD_OUTPUT_METRICS, META_DETECT_METRICS


def center_pd_columns(df):
    for c in df.columns:
        df[c] = df[c] - df[c].mean()

    return df


def center_np_columns(arr):
    for i in range(arr.shape[1]):
        arr[:, i] = arr[:, i] - np.mean(arr[:, i])
    return arr


def standardize_columns(dataframe, columns=None, normalize_critical=False, verbose=False):
    """
    Standardize given columns in dataframe for improved meta model results.
    :param dataframe: (pandas DataFrame) containing the columns to be standardized
    :param columns: (list[str]) containing the column identifier for standardization, e.g. ["xmin", "ymin", "xmax", "ymax", "s"]
    :param normalize_critical: If std of columns happens to be zero, column will only be centered. If True, normalize to [-1, 1] instead.
    :param verbose: Indicated whether error messages will be shown or not.
    :return: dataframe, where the indicated columns were standardized
    """
    std_identifiers = STD_OUTPUT_METRICS + META_DETECT_METRICS + \
        [s for s in dataframe.columns if "grad" in s]

    std_columns = dataframe.columns if columns is None else columns
    for col in std_columns:
        if col in std_identifiers:
            dat = np.copy(np.array(dataframe[col]))
            dat -= np.mean(dat)
            sigma = np.std(dat)
            if sigma != 0.:
                dataframe[col] = dat / sigma
            else:
                if verbose:
                    print(
                        f"Standardization of column '{col}' failed: has standard deviation of {sigma}!")
                if normalize_critical:
                    dat /= np.amax(np.abs(np.copy(dat)))
                    if verbose:
                        print("Data was normalized (kept the sign).")
                else:
                    if verbose:
                        print("Data was only centered.")

    return dataframe
