import os

import pandas as pd

from src.configs.basins import basins, basin_features
from src.configs.regressors import regressors
from src.utils.utils import forward_feature_selection, load_features_target_data


def main():
    """
    Run Forward Feature Selection (FFS) across all basins and model types.

    For each basin:
        - Subset the full features/target dataset to that basin
        - Run FFS for each regressor
            - Track model performance (RRMSE, NSE) at each iteration
            - Track the order in which features are selected

    Output:
        - Individual CSVs for each basin-regressor combination with skill scores (RRMSE, NSE) at each iteration
        - One CSV per basin showing the order in which each feature was selected for each regressor
    """

    data = load_features_target_data()

    for basin in basins:
        print(f'\n=== Processing Basin: {basin} ===')

        # Subset data to current basin
        basin_data = data[data['Basin'] == basin].reset_index(drop=True)
        y = basin_data['Streamflow'].reset_index(drop=True)
        feature_set = basin_features[basin]

        # Initialize a matrix to track feature selection order per regressor
        order_matrix = pd.DataFrame(0, index=feature_set, columns=regressors)

        for regressor in regressors:
            # Perform forward feature selection
            iteration_scores, order = forward_feature_selection(
                basin_data=basin_data,
                y=y,
                feature_set=feature_set,
                regressor=regressor
            )

            # Save the per-iteration scores during FFS (one file per basin/regressor)
            iteration_scores.to_csv(
                path_or_buf=os.path.join('..', 'output', 'forward_feature_selection', f'{basin}_{regressor}.csv')
            )

            # Update order matrix with the iteration at which each feature was selected
            order_matrix[regressor] = order

        # Save feature selection order for all regressors for this basin
        order_matrix.to_csv(
            path_or_buf=os.path.join('..', 'output', 'forward_feature_selection', f'{basin}.csv')
        )


if __name__ == '__main__':
    main()
