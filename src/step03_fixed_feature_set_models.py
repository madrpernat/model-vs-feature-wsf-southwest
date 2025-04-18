import os

import pickle

from src.configs.basins import basins, basin_features
from src.configs.regressors import regressors
from src.utils.data_loaders import load_basin_features_target_data
from src.utils.utils import (calculate_metrics, extract_april_swe_features, init_swe_only_results_dict,
                             nested_cross_validation, set_pipeline_and_param_grid)


def main():
    """
    For each basin, train and evaluate each regressor using only April 1 SWE features, and save results.

    For each basin:
        - Extract April 1 SWE features
        - Load the basin's feature–target dataset and subset to SWE A-only features
        - For each regressor:
            - Run Nested CV using the SWE A-only features
            - Append results to a shared results dictionary

    Output:
        One .pkl file per basin.

        Each .pkl file contains a dictionary with the following keys:
            - 'regressor': The name of the regressor used
            - 'number_of_features': The number of features in the evaluated feature set (i.e., # of SWE A features)
            - 'combo': The list of feature names used in that feature set
            - 'truths': The observed (true) AMJJ water supply values for each test fold
            - 'preds': The corresponding predicted AMJJ water supply values for each test fold
            - 'rrmse_scores': Relative Root Mean Squared Error score for each feature set
            - 'nse_scores': Nash-Sutcliffe Efficiency score for each feature set
            - 'r2_scores': R² score for each feature set
        """

    for basin in basins:
        # Initialize results dictionary for the current basin
        results = init_swe_only_results_dict()

        # Extract only the April 1st SWE features from the current basin's full set of features
        basin_swe_features = extract_april_swe_features(basin_features[basin])

        # Load feature and target data for the current basin and filter features to only the April 1st SWE features
        X, y = load_basin_features_target_data(basin=basin)
        X = X[basin_swe_features]

        for regressor in regressors:
            # Initialize pipeline and hyperparameter grid for the current regressor
            pipe, param_grid = set_pipeline_and_param_grid(regressor)

            # Run Nested CV to get predictions and truth values
            truths, preds = nested_cross_validation(
                pipe=pipe,
                param_grid=param_grid,
                X=X,
                y=y
            )

            # Calculate performance metrics for this SWE A-only model
            rrmse, nse, r2 = calculate_metrics(y_true=truths, y_pred=preds)

            # Record results
            results['regressor'].append(regressor)
            results['number_of_features'].append(len(basin_swe_features))
            results['combo'].append(basin_swe_features),
            results['truths'].append(truths)
            results['preds'].append(preds)
            results['rrmse_scores'].append(rrmse)
            results['r2_scores'].append(r2)
            results['nse_scores'].append(nse)

        # Save results to a single file for the current basin
        with open(
                file=os.path.join(os.getcwd(), '..', 'output', 'swe_only_models', f'{basin}.pkl'),
                mode='wb'
        ) as pickle_file:
            pickle.dump(results, pickle_file)


if __name__ == '__main__':
    main()
