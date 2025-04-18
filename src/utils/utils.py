import itertools
from string import capwords
from typing import List, Tuple, Dict, Union, Any

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.configs.regressors import regressor_titles
from src.utils.data_loaders import load_all_features_target_data, get_best_and_fixed_preds, get_years


def nested_cross_validation(
        pipe: Pipeline,
        param_grid: Union[Dict[str, Any], List[Dict[str, Any]]],
        X: pd.DataFrame,
        y: pd.Series,
        outer_folds: int = 5,
        inner_folds: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Nested Cross-Validation for hyperparameter tuning and return out-of-sample predictions.

    Uses a two-level cross-validation process:
        - Outer Loop: Evaluates generalization performance using `outer_folds` splits
        - Inner Loop: Performs hyperparameter tuning using `inner_folds` splits using GridSearchCV.

    Args:
        pipe (Pipeline): scikit-learn pipeline defining preprocessing and regressor steps.
        param_grid (Union[Dict[str, Any], List[Dict[str, Any]]]): Grid of hyperparameters to tune.
            Accepts either a single dictionary or a list of dictionaries (e.g., for SVR with multiple kernel configs).
        X (pd.DataFrame): Feature DataFrame containing values for each observation year.
        y (pd.Series): The target (AMJJ water supply) for each observation year. Should have the same number of
            observations as `X`.
        outer_folds (int, optional): Number of folds in the outer CV loop. Defaults to 5.
        inner_folds (int, optional): Number of folds in the inner CV loop for GridSearch. Defaults to 5.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Array of true values of outer test folds
            - Array of predicted values across outer test folds
    """
    # Outer CV loop (used for unbiased model performance estimation)
    kf = KFold(n_splits=outer_folds, shuffle=True, random_state=123)
    preds, truths = [], []

    for train_idx, test_idx in kf.split(X):
        # Split into training and testing sets for the current outer fold
        X_train, X_test = X.loc[train_idx, :], X.loc[test_idx, :]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]

        # Inner loop: grid search over hyperparameter grid to find best hyperparameters
        optimized_regressor = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=inner_folds,  # inner loop
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )

        # Using the best hyperparameters, fit model on training data and predict on the outer fold test data
        optimized_regressor.fit(X_train, y_train)
        test_fold_preds = optimized_regressor.predict(X_test)

        # Store prediction and truth values
        preds.extend(test_fold_preds)
        truths.extend(y_test.to_list())

    return np.array(truths), np.array(preds)


def forward_feature_selection(
        basin_data: pd.DataFrame,
        y: pd.Series,
        feature_set: List[str],
        regressor: str,
        n_features: int = 10
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Perform forward feature selection for one regressor and one basin.

    At each step, the feature that yields the best performance (based on RRMSE) is added to the `selected` feature list.
    Model performance is evaluated using Nested Cross-Validation.

    Args:
        basin_data (pd.DataFrame): A DataFrame containing feature values for each observation year.
        y (pd.Series): The target (AMJJ water supply) for each observation year. Should have the same number of
            observations as `basin_data`.
        feature_set (List[str]): List of candidate feature names to consider during selection. Each must match a column
            name in `basin_data`.
        regressor (str): Name of the regressor to use. Options include:
            'LinearRegression', 'PrincipalComponentRegression', 'SVR', 'RandomForestRegressor', 'ExtraTreesRegressor'
        n_features (int, optional): Number of features to select. Defaults to 10.

    Returns:
        Tuple[pd.DataFrame, pd.Series]:
            - iteration_scores (pd.DataFrame): Model skill at each iteration (rows). Columns include rrmse, nse, and r2.
            - order (pd.Series): A Series where the index contains feature names and the values indicate the order in
                which each feature was selected (1 = selected first, 2 = selected second, ..., 0 = not selected).
    """
    # Set up pipeline and hyperparameter grid for the selected regressor
    pipe, param_grid = set_pipeline_and_param_grid(regressor_name=regressor)

    # Start with no selected features; all features are initially unselected and available for evaluation
    selected, remaining = [], feature_set.copy()

    # Initialize dataframe to store performance metrics at each iteration
    iteration_scores = pd.DataFrame(index=range(1, n_features + 1), columns=['rrmse', 'nse', 'r2'])

    # Initialize Series to store selection order of each feature (0 = not selected)
    order = pd.Series(0, index=feature_set)

    for i in range(1, n_features + 1):
        metrics = []

        # Evaluate the impact of adding each remaining feature
        for feature in remaining:
            current_set = selected + [feature]

            # Filter data matrix to current feature set
            X = basin_data[current_set]

            # Perform Nested CV to get out-of-sample predictions
            truths, preds = nested_cross_validation(
                pipe=pipe,
                param_grid=param_grid,
                X=X,
                y=y
            )

            # Calculate performance metrics for the current feature set
            rrmse, nse, r2 = calculate_metrics(y_true=truths, y_pred=preds)
            metrics.append((feature, rrmse, nse, r2))

            print(f'{regressor} | Features: {current_set} | RRMSE: {rrmse:.3f}')

        # Select the feature that results in the best (lowest) RRMSE
        best_feature, best_rrmse, best_nse, best_r2 = min(metrics, key=lambda x: x[1])
        selected.append(best_feature)
        remaining.remove(best_feature)

        # Record selection order and model performance at this iteration
        order[best_feature] = i
        iteration_scores.loc[i] = [best_rrmse, best_nse, best_r2]

    return iteration_scores, order


def exhaustive_search(
        basin_data: pd.DataFrame,
        y: pd.Series,
        top_features: List[str],
        regressor: str
) -> Dict[str, List[Any]]:
    """
    Perform exhaustive feature search of all possible feature sets in the provided `top_features`.
        - Generate all possible non-empty sets of `top_features`
        - Evaluate each set using Nested CV
        - Store predictions, truth values, and skill scores (RRMSE, NSE, R2)

    Args:
        basin_data (pd.DataFrame): A DataFrame containing feature values for each observation year. Should include
            all columns referenced in `top_features`.
        y (pd.Series): The target (AMJJ water supply) for each observation year. Should have the same number of
            observations as `basin_data`.
        top_features (List[str]): Features to evaluate within the exhaustive search,  selected via forward feature
            selection.
        regressor (str): Name of the regressor to evaluate. Options include:
            'LinearRegression', 'PrincipalComponentRegression', 'SVR', 'RandomForestRegressor', 'ExtraTreesRegressor'

    Returns:
        Dict[str, List[Any]]: Dictionary containing predictions, truth values, feature names,
            and skill metrics for each feature set evaluated within the exhaustive search.
    """
    results = init_exhaustive_search_results_dict()

    # Set up pipeline and hyperparameter grid for the selected regressor
    pipe, param_grid = set_pipeline_and_param_grid(regressor_name=regressor)

    # Generate all (non-empty) sets of features in `top_features`
    all_combos = create_all_sets(features_list=top_features)

    for i, combo in enumerate(all_combos):
        print(f'    Combo {i + 1}/{len(all_combos)}: {combo}')

        # Filter data matrix to current feature set
        X = basin_data[combo].reset_index(drop=True)

        # Perform Nested CV to get out-of-sample predictions
        truths, preds = nested_cross_validation(
            pipe=pipe,
            param_grid=param_grid,
            X=X,
            y=y
        )

        # Calculate performance metrics for the current feature set
        rrmse, nse, r2 = calculate_metrics(y_true=truths, y_pred=preds)

        # Record results
        results['regressor'].append(regressor)
        results['number_of_features'].append(len(combo))
        results['combo'].append(combo)
        results['truths'].append(truths)
        results['preds'].append(preds)
        results['rrmse_scores'].append(rrmse)
        results['nse_scores'].append(nse)
        results['r2_scores'].append(r2)

    return results


def set_pipeline_and_param_grid(regressor_name: str) -> Tuple[Pipeline, Union[Dict[str, Any], List[Dict[str, Any]]]]:
    """
    Set up a scikit-learn pipeline and corresponding hyperparameter grid for the specified regressor.

    Supported regressors:
        - 'LinearRegression': No hyperparameters to tune
        - 'PrincipalComponentRegression': Linear regression on PCA-transformed features;
          tunes the number of PCA components (as a proportion of explained variance)
        - 'SVR': Support Vector Regression with multiple kernel options (linear, poly, rbf, sigmoid)
        - 'RandomForestRegressor': Random forest with number of trees, max depth, and max features tuning
        - 'ExtraTreesRegressor': Extra-trees ensemble with number of trees, max depth, and max features tuning

    Args:
        regressor_name (str): Name of the regressor to configure.

    Returns:
        Tuple[Pipeline, Union[Dict[str, Any], List[Dict[str, Any]]]]:
            - A scikit-learn Pipeline with preprocessing and regressor
            - A hyperparameter grid for use with GridSearchCV
        """
    # Multiple Linear Regression: no hyperparameters to tune
    if regressor_name == 'LinearRegression':
        pipe = Pipeline([
            ('transformer', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        param_grid = {}

    # Principal Component Regression: tune number of PCA components (as proportion of variance retained)
    elif regressor_name == 'PrincipalComponentRegression':
        pipe = Pipeline([
            ('transformer', StandardScaler()),
            ('pca', PCA()),
            ('regressor', LinearRegression())
        ])
        param_grid = {'pca__n_components': [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]}

    # Support Vector Regression: tune epsilon, C, kernel function, and kernel-specific parameters
    elif regressor_name == 'SVR':

        pipe = Pipeline([
            ('transformer', StandardScaler()),
            ('regressor', SVR())
        ])
        param_grid = [
            {  # 'poly' kernel: includes 'degree' and 'gamma'
                'regressor__C': np.logspace(-1, 3, num=10).tolist(),
                'regressor__epsilon': np.logspace(-2, 0, num=5).tolist(),
                'regressor__gamma': ['scale', 'auto'],
                'regressor__kernel': ['poly'],
                'regressor__degree': [2, 3, 4]
            },
            {  # 'rbf' and 'sigmoid' kernels: include 'gamma', exclude 'degree'
                'regressor__C': np.logspace(-1, 3, num=10).tolist(),
                'regressor__epsilon': np.logspace(-2, 0, num=5).tolist(),
                'regressor__gamma': ['scale', 'auto'],
                'regressor__kernel': ['rbf', 'sigmoid']
            },
            {  # 'linear' kernel: excludes both 'gamma' and 'degree'
                'regressor__C': np.logspace(-1, 3, num=10).tolist(),
                'regressor__epsilon': np.logspace(-2, 0, num=5).tolist(),
                'regressor__kernel': ['linear']
            }
        ]

    # Random Forest: tune number of trees, depth, and max feature sampling
    elif regressor_name == 'RandomForestRegressor':
        pipe = Pipeline([
            ('transformer', StandardScaler()),
            ('regressor', RandomForestRegressor())
        ])
        param_grid = {
            'regressor__n_estimators': [5, 10, 25, 50, 75, 100, 125],
            'regressor__max_depth': [2, 3, 4, 5, None],
            'regressor__max_features': [0.25, 0.5, 0.75, None, 'sqrt']
        }

    # Extremely Randomized Trees: tune number of trees, depth, and max feature sampling
    elif regressor_name == 'ExtraTreesRegressor':
        pipe = Pipeline([
            ('transformer', StandardScaler()),
            ('regressor', ExtraTreesRegressor(
            ))
        ])
        param_grid = {
            'regressor__n_estimators': [5, 10, 25, 50, 75, 100, 125],
            'regressor__max_depth': [2, 3, 4, None],
            'regressor__max_features': [0.25, 0.5, 0.75, None, 'sqrt']
        }

    return pipe, param_grid


def calc_nse(
        y_true: np.array,
        y_pred: np.array
) -> float:
    """
    Calculate Nash-Sutcliffe Efficiency between true and predicted values.

    Args:
        y_true (np.array): Observed values.
        y_pred (np.array): Predicted values.

    Returns:
        float: NSE
    """
    residuals = np.sum((y_true - y_pred) ** 2)
    variance = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - residuals / variance


def calc_rrmse(
        y_true: np.array,
        y_pred: np.array
) -> float:
    """
    Calculate Root Relative Mean Squared Error (RRMSE) and R² between true and predicted values.

    RRMSE is the RMSE normalized by the mean of the true values.

    Args:
        y_true (np.array): Observed values.
        y_pred (np.array): Predicted values.

    Returns:
        Tuple[float, float]: (RRMSE, R²)
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2)) / np.mean(y_true)


def calc_r2(
        y_true: np.array,
        y_pred: np.array
) -> float:
    """
    Calculate Root Relative Mean Squared Error (RRMSE) and R² between true and predicted values.

    RRMSE is the RMSE normalized by the mean of the true values.

    Args:
        y_true (np.array): Observed values.
        y_pred (np.array): Predicted values.

    Returns:
        Tuple[float, float]: (RRMSE, R²)
    """
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    return float(correlation ** 2)


def calculate_metrics(
        y_true: np.array,
        y_pred: np.array
) -> Tuple[float, float, float]:
    """
    Calculate RRMSE, NSE, and R² between true and predicted values. Wrapper around `calc_rrmse` and `calc_r2` and
    `calc_nse` for convenience.

    Args:
        y_true (np.array): Array of observed values.
        y_pred (np.array): Array of predicted values.

    Returns:
        Tuple[float, float, float]: (RRMSE, NSE, R²)
    """
    rrmse = calc_rrmse(y_true=y_true, y_pred=y_pred)
    r2 = calc_r2(y_true=y_true, y_pred=y_pred)
    nse = calc_nse(y_true=y_true, y_pred=y_pred)
    return rrmse, nse, r2


def create_all_sets(features_list: list[str]) -> List[List[str]]:
    """
    Generate all non-empty subsets of a given list of features.

    Args:
        features_list (List[str]): List of feature names.

    Returns:
        List[List[str]]: All non-empty combinations of the list of features.
    """
    combos = []
    for L in range(len(features_list) + 1):
        for subset in itertools.combinations(features_list, L):
            combos.append(list(subset))
    return combos[1:]


def extract_april_swe_features(feature_list: List[str]) -> List[str]:
    """
    Extract April 1 SWE features from a full list of  features.

    Args:
        feature_list (List[str]): List of candidate feature names.

    Returns:
        List[str]: Subset of feature names that include 'SWE_A' (April 1 SWE).
    """
    return [item for item in feature_list if 'SWE_A' in item]


def init_exhaustive_search_results_dict() -> Dict[str, List[Any]]:
    """
    Initialize a results dictionary to store exhaustive feature search outputs.

    Returns:
        Dict[str, List[Any]]: A dictionary with empty lists for each result field.
    """
    return {
        'regressor': [],
        'number_of_features': [],
        'combo': [],
        'truths': [],
        'preds': [],
        'rrmse_scores': [],
        'r2_scores': [],
        'nse_scores': [],
    }


def init_swe_only_results_dict() -> Dict[str, List[Any]]:
    """
    Initialize a results dictionary for SWE-only model evaluation. Same as for `init_exhaustive_search_results_dict()`.
    """
    return init_exhaustive_search_results_dict()


def lolipop_plot(ax, regressors, data_a, data_b, color_a, color_b, label_a, label_b, ylims, yticks, invert_y=False):

    xpositions = np.arange(len(regressors))
    xlabels = [regressor_titles[regressor] for regressor in regressors]

    mins = [min([i, j]) for i, j in zip(data_a, data_b)]
    maxs = [max([i, j]) for i, j in zip(data_a, data_b)]

    ax.vlines(
        x=xpositions,
        ymin=mins,
        ymax=maxs,
        color='black',
        alpha=0.8,
        zorder=2
    )
    ax.scatter(
        x=xpositions,
        y=data_a,
        color=color_a,
        label=label_a,
        zorder=3
    )
    ax.scatter(
        x=xpositions,
        y=data_b,
        color=color_b,
        label=label_b,
        zorder=3
    )
    ax.set_xticks(xpositions)
    ax.set_xticklabels(labels=xlabels, ha='center')
    ax.set_ylim(ylims)
    ax.set_yticks(yticks)

    if invert_y:
        ax.invert_yaxis()

    ax.grid(
        visible=True,
        which='both',
        axis='y',
        linestyle='--',
        alpha=0.3,
        zorder=1
    )
    add_scatter_values(ax=ax, xpositions=xpositions, mins=mins, maxs=maxs, invert=invert_y)


def add_scatter_values(ax, xpositions, mins, maxs, invert):

    if not invert:
        ymin, ymax = ax.get_ylim()
    else:
        ymin, ymax = (abs(ax.get_ylim()[1]), abs(ax.get_ylim()[0]))
    yrange = ymax - ymin

    # Max labels
    for x, y in zip(xpositions, maxs):

        if (ymax - y) / yrange < 0.17:
            x = x + 0.26
            yposition = y - 0.02 * yrange if not invert else y + 0.16 * yrange
        else:
            x = x
            yposition = y + 0.025 * yrange if not invert else y + 0.13 * yrange
        ax.text(x=x, y=yposition, s=f'{y:.2f}', ha='center', va='bottom', fontsize=12)

    # Min labels
    for x, y in zip(xpositions, mins):

        if (y - ymin) / yrange < 0.05:
            x = x + 0.26
            yposition = y + 0.07 * yrange if not invert else y + 0.01 * yrange
        elif (y - ymin) / yrange < 0.15:
            x = x + 0.26
            yposition = y + 0.07 * yrange if not invert else y - 0.06 * yrange
        else:
            x = x
            yposition = y - 0.05 * yrange if not invert else y - 0.11 * yrange
        ax.text(x=x, y=yposition, s=f'{y:.2f}', ha='center', va='top', fontsize=12)


def reorder(basin, truths, best_preds, swe_preds):

    # Original DataFrame
    data = load_all_features_target_data()
    basin_data = data[data['Basin'] == basin][['Year', 'Streamflow']].sort_values(by='Year', ascending=True)

    # Unordered truths/preds
    unordered_streamflow = pd.DataFrame({'truths': truths, 'best_preds': best_preds, 'swe_preds': swe_preds})

    # Create a mapping from Streamflow to Year from the original df
    sf_to_year = dict(zip(basin_data['Streamflow'], basin_data['Year']))

    # Add corresponding Year using the mapping
    unordered_streamflow['Year'] = unordered_streamflow['truths'].map(sf_to_year)

    # (Optional) Sort by Year
    ordered_preds = unordered_streamflow.sort_values(by='Year', ascending=True).reset_index(drop=True)

    return ordered_preds


def timeseries_plot(ax, basin, regressor, yticks):
    truths, best_preds, swe_preds = get_best_and_fixed_preds(basin=basin, regressor=regressor)
    df = reorder(basin=basin, truths=truths, best_preds=best_preds, swe_preds=swe_preds)
    years = get_years()

    ax.plot(years, df['truths'], color='black', label='Observations', linestyle='-', marker='o', markersize=4)
    ax.plot(years, df['best_preds'], color='red', label='Best Feature Set', linestyle='--', marker='o', markersize=4)
    ax.plot(years, df['swe_preds'], color='blue', label='Fixed Feature Set (SWE A)', linestyle='--', marker='o', markersize=4)

    ax.set_xlabel('Year')
    ax.set_ylabel('AMJJ Water\nSupply (cfs)', color='black')
    ax.set_yticks(yticks)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim(min(years) - 1, max(years) + 1)
    ax.set_title(capwords(basin), fontsize=16)
