import os

import pickle

from src.configs.basins import basins
from src.configs.regressors import regressors
from src.utils.utils import exhaustive_search, init_exhaustive_search_results_dict
from src.utils.data_loaders import get_ffs_order, load_basin_features_target_data


def main():
    """
    For each basin, and for each regressor, perform an exhaustive feature search using the top 10 features identified
    through forward feature selection, and save the results.

    For each basin:
        - Load the basin's feature–target dataset
        - Load the feature selection order matrix from FFS
        - For each regressor:
            - Extract the top 10 features selected by FFS (features with rank > 0)
            - Run an exhaustive search over all non-empty combinations of those features using `exhaustive_search`,
              which returns predictions, true values, and skill metrics (RRMSE, NSE, R²) for each feature set
            - Append results to a shared `basin_results` dictionary

    Output:
        One .pkl file per basin.

        Each .pkl file contains a dictionary with the following keys:
            - 'regressor': The name of the regressor used
            - 'number_of_features': The number of features in the evaluated feature set
            - 'combo': The list of feature names used in that feature set
            - 'truths': The observed (true) AMJJ water supply values for each test fold
            - 'preds': The corresponding predicted AMJJ water supply values for each test fold
            - 'rrmse_scores': Relative Root Mean Squared Error score for each feature set
            - 'nse_scores': Nash-Sutcliffe Efficiency score for each feature set
            - 'r2_scores': R² score for each feature set
    """
    for basin in basins:
        print(f'\n=== Processing Basin: {basin} ===')

        # Load feature and target data
        X, y = load_basin_features_target_data(basin=basin)

        # Load FFS order matrix
        ffs_order = get_ffs_order(basin)

        # Initialize a results dictionary for this basin
        basin_results = init_exhaustive_search_results_dict()

        for regressor in regressors:
            print(f'\n--- Regressor: {regressor} ---')

            # Extract the selected features for the current regressor
            top_features = ffs_order[(ffs_order[regressor] > 0)].index.tolist()

            # Run exhaustive search over all non-empty sets of top features
            regressor_results = exhaustive_search(
                basin_data=X,
                y=y,
                top_features=top_features,
                regressor=regressor
            )

            # Append regressor-specific results to the shared basin dictionary
            for key in basin_results:
                basin_results[key].extend(regressor_results[key])

        # Save results to output dictionary
        out_path = os.path.join('..', 'output', 'exhaustive_feature_search', f'{basin}.pkl')
        with open(file=out_path, mode='wb') as pickle_file:
            pickle.dump(basin_results, pickle_file)


if __name__ == '__main__':
    main()
