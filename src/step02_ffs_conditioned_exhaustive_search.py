import os

import pickle

from src.configs.basins import basins
from src.configs.regressors import regressors
from src.utils.utils import (exhaustive_search, get_ffs_order, init_exhaustive_search_results_dict,
                             load_features_target_data)


def main():

    data = load_features_target_data()

    for basin in basins:
        print(f'\n=== Processing Basin: {basin} ===')

        # Subset data to the current basin
        basin_data = data[data['Basin'] == basin].reset_index(drop=True)
        y = basin_data['Streamflow'].reset_index(drop=True)

        # Load FFS order matrix
        ffs_order = get_ffs_order(basin)

        # Initialize a shared results dictionary for this basin
        basin_results = init_exhaustive_search_results_dict()

        for regressor in regressors:
            print(f'\n--- Regressor: {regressor} ---')

            # Extract the selected features for the current regressor
            top_features = ffs_order[(ffs_order[regressor] > 0)].index.tolist()

            # Run exhaustive search over all non-empty sets of top features
            regressor_results = exhaustive_search(
                basin_data=basin_data,
                y=y,
                top_features=top_features,
                regressor=regressor
            )

            # Append regressor-specific results to the shared basin dictionary
            for key in basin_results:
                basin_results[key].extend(regressor_results[key])

        # Save results to output dictionary
        out_path = os.path.join('..', 'output', 'exhaustive_feature_search', f'{basin}.pkl')
        with open(file=out_path,mode='wb') as pickle_file:
            pickle.dump(basin_results, pickle_file)


if __name__ == '__main__':
    main()
