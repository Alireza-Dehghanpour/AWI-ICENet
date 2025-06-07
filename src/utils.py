from sklearn import metrics
import math
import pandas as pd

def get_metrics(y_true, y_pred, data_pd, row_idx, fold_num):
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)


    mse = round(mse, 4)
    rmse = round(rmse, 4)
    mae = round(mae, 4)
    r2 = round(r2, 4)

    data_pd.iloc[row_idx, 1] = mse
    data_pd.iloc[row_idx, 2] = rmse
    data_pd.iloc[row_idx, 3] = mae
    data_pd.iloc[row_idx, 4] = r2
    data_pd.iloc[row_idx, 5] = int(fold_num)

    return data_pd

def init_results_dataframe(algorithm_name, cv_folds):
    zeros_data = [[0]*6 for _ in range(len(algorithm_name) * cv_folds)]
    columns = ["algorithm_name", "MSE", "RMSE", "MAE", "R2", "k"]
    df = pd.DataFrame(zeros_data, columns=columns)
    df["algorithm_name"] = algorithm_name * cv_folds
    return df

