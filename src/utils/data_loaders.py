import glob
import os
from string import capwords
from typing import List, Tuple

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


def get_ffs_iteration_scores(basin: str, regressor: str) -> pd.DataFrame:
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


def get_exhaustive_search_results(basin: str) -> pd.DataFrame:
    """
    Load exhaustive search results for a given basin (generated from `step02_ffs_conditioned_exhaustive_search.py).

    Args:
        basin (str): Basin name.

    Returns:
        pd.DataFrame: Exhaustive search results as a flat DataFrame.
    """
    with open(os.path.join('..', 'output', 'exhaustive_feature_search', f'{basin}.pkl'), 'rb') as file:
        results = pickle.load(file)
    return pd.DataFrame(results)


def get_swe_only_results(basin: str) -> pd.DataFrame:
    """
    Load SWE-only model results for a given basin (generated from `step03_fixed_feature_set_models.py).

    Args:
        basin (str): Basin name.

    Returns:
        pd.DataFrame: SWE-only results as a flat DataFrame.
    """
    with open(os.path.join('..', 'output', 'swe_only_models', f'{basin}.pkl'), 'rb') as file:
        results = pickle.load(file)
    return pd.DataFrame(results)


def get_streamflow_data(basin: str) -> pd.DataFrame:
    """
    Load streamflow observations CSV for a given basin.

    Args:
        basin (str): Basin name.

    Returns:
        pd.DataFrame: DataFrame containing streamflow timeseries.
    """
    parent_dir = os.path.join(os.getcwd(), '..', 'data', 'streamflow')
    parent_dir = os.path.abspath(parent_dir)
    file_pattern = os.path.join(parent_dir, f'{capwords(basin)}_*.csv')
    target_file = glob.glob(file_pattern)[0]
    return pd.read_csv(target_file)


def get_years() -> List[int]:
    """
    Get the sorted list of years from the full feature-target dataset.

    Returns:
        List[int]: Unique years present in the dataset.
    """
    data = load_all_features_target_data()
    years = data['Year'].unique()
    return sorted(years)


def get_best_and_fixed_scores(
        regressors: List[str],
        basin: str
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    For a given basin, retrieve the 'best' and 'fixed' skill scores for each regressor.

    For each regressor:
        - Select the lowest (best) RRMSE model from the exhaustive search
        - Select the SWE-only model
        - Extract RRMSE and NSE for both

    Args:
        regressors (List[str]): List of regressors.
        basin (str): Basin name.

    Returns:
        Tuple[List[float], List[float], List[float], List[float]]:
            - List of best RRMSE values (one per regressor)
            - List of best NSE values (one per regressor)
            - List of SWE-only RRMSE values (one per regressor)
            - List of SWE-only NSE values (one per regressor)
    """
    # Load exhaustive search and swe-only results
    exhaustive = get_exhaustive_search_results(basin)
    swe_only = get_swe_only_results(basin)

    # Initialize output lists
    best_rrmse, best_nse = [], []
    swe_rrmse, swe_nse = [], []

    for regressor in regressors:
        # Select the best model (lowest RRMSE) from exhaustive search
        best_model = (
            exhaustive[exhaustive['regressor'] == regressor]
            .sort_values(by='rrmse_scores')
            .iloc[0]
        )

        # Select the SWE-only model for the same regressor
        swe_model = swe_only[swe_only['regressor'] == regressor].iloc[0]

        # Extract skill scores
        best_rrmse.append(best_model['rrmse_scores'])
        best_nse.append(best_model['nse_scores'])
        swe_rrmse.append(swe_model['rrmse_scores'])
        swe_nse.append(swe_model['nse_scores'])

    return best_rrmse, best_nse, swe_rrmse, swe_nse


def get_best_and_fixed_preds(
        basin: str,
        regressor: str
) -> Tuple[List[float], List[float], List[float]]:
    """
    Retrieve prediction values from the best model (from exhaustive search) and
    the SWE-only model for a given basin and regressor.

    For the given regressor:
        - Select the model with the lowest RRMSE from exhaustive search
        - Select the corresponding SWE-only model
        - Return predictions from both, along with the shared true values

    Args:
        basin (str): Name of the basin.
        regressor (str): Name of the regressor.

    Returns:
        Tuple[List[float], List[float], List[float]]:
            - True AMJJ streamflow values
            - Predictions from the best model in exhaustive search
            - Predictions from the SWE-only model
    """
    # Load results
    exhaustive = get_exhaustive_search_results(basin)
    swe_only = get_swe_only_results(basin)

    # Filter to the specified regressor
    exhaustive = exhaustive[exhaustive['regressor'] == regressor]
    swe_only = swe_only[swe_only['regressor'] == regressor]

    # Select the best model (lowest RRMSE) from exhaustive search
    best_model = exhaustive.sort_values(by='rrmse_scores').iloc[0]

    # Select the SWE-only model (only one per regressor)
    swe_model = swe_only.iloc[0]

    # Return truths, best model preds, and SWE-only preds
    return best_model['truths'], best_model['preds'], swe_model['preds']
