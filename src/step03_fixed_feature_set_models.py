import os
from typing import List

import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV, KFold

from src.configs.basin_names import basins
from src.configs.basin_features import basin_features
from src.configs.regressors import regressors
from src.utils.utils import calc_nse, calc_rrmse_r2, load_features_target_data, set_pipeline_and_param_grid


def extract_april_swe_features(feature_list: List[str]):
    return [item for item in feature_list if 'SWE_A' in item]


def init_swe_only_results_dict():
    return {
        'regressor': [], 'number_of_features': [], 'combo': [], 'truths': [], 'preds': [], 'rrmse_scores': [],
        'r2_scores': [], 'nse_scores': []
    }


def main():

    data = load_features_target_data()

    for basin in basins:

        results = init_swe_only_results_dict()

        basin_data = data[data['Basin'] == basin]
        basin_swe_features = extract_april_swe_features(basin_features[basin])

        X = basin_data[basin_swe_features].reset_index(drop=True)
        y = basin_data['Streamflow'].reset_index(drop=True)

        for regressor in regressors:

            pipe, param_grid = set_pipeline_and_param_grid(regressor)

            # Repeated k-fold instance
            kf = KFold(n_splits=5, shuffle=True, random_state=123)
            preds, truths = [], []

            for _, (train_idx, test_idx) in enumerate(kf.split(X)):
                X_train, X_test = X.loc[train_idx, :], X.loc[test_idx, :]
                y_train, y_test = y.loc[train_idx], y.loc[test_idx]

                optimized_regressor = GridSearchCV(
                    estimator=pipe,
                    param_grid=param_grid,
                    cv=5,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1
                )

                optimized_regressor.fit(X_train, y_train)
                pred = optimized_regressor.predict(X_test)
                preds.extend(pred)
                truths.extend(y_test.to_list())

            # Calculate test metrics
            nse = calc_nse(y_true=np.array(truths), y_pred=np.array(preds))
            rrmse, r2 = calc_rrmse_r2(y_true=truths, y_pred=preds)

            results['regressor'].append(regressor)
            results['number_of_features'].append(len(basin_swe_features))
            results['combo'].append(basin_swe_features),
            results['truths'].append(truths)
            results['preds'].append(preds)
            results['rrmse_scores'].append(rrmse)
            results['r2_scores'].append(r2)
            results['nse_scores'].append(nse)

        with open(
                file=os.path.join(os.getcwd(), '..', 'output', 'swe_only_models', f'{basin}_svr_updated.pkl'),
                mode='wb'
        ) as pickle_file:
            pickle.dump(results, pickle_file)


if __name__ == '__main__':
    main()
