import os
from typing import Tuple

import pandas as pd
import pickle


def load_all_features_target_data() -> pd.DataFrame:
    """
    Load the full dataset containing both features and target variable for all basins
    """
    return pd.read_csv(os.path.join('..', 'data', 'full_feature_target_data.csv'))


def load_basin_features_target_data(basin: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load a basin's feature and target data.

    Args:
        basin (str): Name of the basin.

    Returns:
        pd.DataFrame: Basin's feature data
        pd.Series: Basin's target data
    """
    data = load_all_features_target_data()

    basin_data = data[data['Basin'] == basin].reset_index(drop=True)
    y = basin_data['Streamflow'].reset_index(drop=True)

    return basin_data, y


def get_ffs_order(basin: str) -> pd.DataFrame:
    """
    Load the feature selection order matrix for a given basin.

    Args:
        basin (str): Name of the basin.

    Returns:
        pd.DataFrame: Feature selection order matrix.
    """
    return pd.read_csv(
        os.path.join('..', 'output', 'forward_feature_selection', f'{basin}.csv'),
        index_col=0
    )


def get_ffs_iteration_scores(basin: str, regressor: str):
    """
    Load the per-iteration skill scores from forward feature selection for a given basin and regressor. Each row
    corresponds to an iteration, and columns include RRMSE, NSE, and R2.

    Args:
        basin (str): Name of the basin.
        regressor (str): Name of the regressor.

    Returns:
        pd.DataFrame: Iteration-level performance scores.
    """

    return pd.read_csv(
        os.path.join('..', 'output', 'forward_feature_selection', f'{basin}_{regressor}.csv'),
        index_col=0
    )


def get_exhaustive_search_results(basin: str):
    with open(os.path.join('..', 'output', 'exhaustive_feature_search', f'{basin}.pkl'), 'rb') as file:
        results = pickle.load(file)
    return pd.DataFrame(results)


def get_swe_only_results(basin: str):
    with open(os.path.join('..', 'output', 'swe_only_models', f'{basin}.pkl'), 'rb') as file:
        results = pickle.load(file)
    return pd.DataFrame(results)


def get_streamflow_data(basin: str):
    parent_dir = os.path.join(os.getcwd(), '..', 'data', 'streamflow')
    parent_dir = os.path.abspath(parent_dir)
    file_pattern = os.path.join(parent_dir, f'{capwords(basin)}_*.csv')
    target_file = glob.glob(file_pattern)[0]
    return pd.read_csv(target_file)


def get_years():
    data = load_all_features_target_data()
    years = data['Year'].unique()
    return sorted(years)


def get_best_and_fixed_scores(regressors, basin):

    exhaustive_search_results = get_exhaustive_search_results(basin)
    swe_only_results = get_swe_only_results(basin)

    best_rrmse, best_nse = [], []
    swe_rrmse, swe_nse = [], []

    for regressor in regressors:
        best_model = exhaustive_search_results[
            exhaustive_search_results['regressor'] == regressor
            ].sort_values(by='rrmse_scores', ascending=True).iloc[0]

        best_rrmse.append(best_model['rrmse_scores'])
        best_nse.append(best_model['nse_scores'])
        swe_rrmse.append(
            swe_only_results.at[swe_only_results[swe_only_results['regressor'] == regressor].index[0], 'rrmse_scores'])
        swe_nse.append(
            swe_only_results.at[swe_only_results[swe_only_results['regressor'] == regressor].index[0], 'nse_scores'])

    return best_rrmse, best_nse, swe_rrmse, swe_nse


def get_best_and_fixed_preds(basin, regressor):

    exhaustive_search_results = get_exhaustive_search_results(basin)
    swe_only_results = get_swe_only_results(basin)

    # Filter to regressor
    exhaustive_search_results = exhaustive_search_results[exhaustive_search_results['regressor'] == regressor]
    swe_only_results = swe_only_results[swe_only_results['regressor'] == regressor]

    best_model = exhaustive_search_results.sort_values(by='rrmse_scores', ascending=True).iloc[0]
    swe_model = swe_only_results.sort_values(by='rrmse_scores', ascending=True).iloc[0]

    return best_model['truths'], best_model['preds'], swe_model['preds']