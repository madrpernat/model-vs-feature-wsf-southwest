import os
from string import capwords

import matplotlib
import matplotlib.pyplot as plt

from src.configs.basins import basins
from src.configs.features import acronyms, colors
from src.configs.regressors import regressors, regressor_titles
from src.utils.data_loaders import get_exhaustive_search_results, get_ffs_order


matplotlib.use('Qt5Agg')
plt.rcParams.update({
    'font.size': 16,
    'font.family': 'Times New Roman',
    'axes.titlesize': 17,
    'axes.labelsize': 14,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16
})


def main():

    ffs_features_lists = []
    best_features_lists = []

    for basin in basins:

        ffs_results = get_ffs_order(basin)
        exhaustive_search_results = get_exhaustive_search_results(basin)

        for regressor in regressors:

            # For the given basin/model, get the 10 features that were selected by FFS
            ffs_features = [ffs_results[ffs_results[regressor] == i].index[0] for i in range(1, 11)]
            ffs_features_lists.append(ffs_features)

            # For the given basin/model, get the features present in the best feature set. Compare to the FFS set.
            best_model = exhaustive_search_results[
                             exhaustive_search_results['regressor'] == regressor
                             ].sort_values(by='rrmse_scores', axis=0, ascending=True).iloc[0, :]

            best_features = best_model['combo']
            best_features_lists.append(best_features)

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 13))
    ax.axis('off')
    y_labels = [f"{capwords(basin)} - {regressor_titles[regressor]}" for basin in basins for regressor in regressors]

    # Create colors: highlight specific cells
    cell_colors = []
    for ffs_features, best_features in zip(ffs_features_lists, best_features_lists):
        row_colors = []
        for feature in ffs_features:
            if feature in best_features:
                row_colors.append(colors[feature])
            else:
                row_colors.append('white')
        cell_colors.append(row_colors)

    # Create table cell text
    cell_text = [[acronyms[feature] for feature in sublist] for sublist in ffs_features_lists]

    # --- Lower layer: color + text (no borders) ---
    colors_table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        colLabels=[f'Iteration {i + 1}' for i in range(10)],
        rowLabels=y_labels,
        cellLoc='center',
        bbox=[0.0, 0.0, 1.0, 1.0],
        zorder=1
    )

    # --- Upper layer: borders only (transparent text and background) ---
    edges_table = ax.table(
        cellText=cell_text,
        colLabels=[f'Iteration {i + 1}' for i in range(10)],
        rowLabels=y_labels,
        cellLoc='center',
        bbox=[0.0, 0.0, 1.0, 1.0],
        zorder=10
    )

    # Make upper table transparent (but keep borders)
    for cell in edges_table.get_celld().values():
        cell.set_facecolor((1, 1, 1, 0))  # transparent background
        cell.get_text().set_color((1, 1, 1, 0))  # hide text

    # Apply custom border logic
    n_rows = len(cell_text)
    n_cols = len(cell_text[0])

    for row in range(n_rows + 1):  # includes header
        for col in range(-1, n_cols):  # include row label column
            # Skip the (0, -1) cell — it doesn't exist
            if (row, col) == (0, -1):
                continue
            cell = edges_table[row, col]
            edges = 'B'
            cell.visible_edges = edges
            cell.set_linewidth(0.5)

    # Remove borders from lower layer
    for cell in colors_table.get_celld().values():
        cell.set_linewidth(0)

    # --- Third layer: THICK lines only ---
    thicklines_table = ax.table(
        cellText=cell_text,
        colLabels=[f'Iteration {i + 1}' for i in range(10)],
        rowLabels=y_labels,
        cellLoc='center',
        bbox=[0.0, 0.0, 1.0, 1.0],
        zorder=15  # Between color and edge layer, or above both if needed
    )

    # Make the thicklines_table transparent (no text, no fill)
    for cell in thicklines_table.get_celld().values():
        cell.set_facecolor((1, 1, 1, 0))
        cell.get_text().set_color((1, 1, 1, 0))
        cell.visible_edges = ''

    # Add thick horizontal line every 5 rows
    for row in range(0, n_rows + 1, 5):  # includes header at row 0
        for col in range(-1, n_cols):
            if (row, col) == (0, -1): continue  # skip if non-existent
            cell = thicklines_table[row, col]
            cell.visible_edges += 'B'
            cell.set_linewidth(2.0)

    # Add thick vertical line between col 3 and 4
    for row in range(1, n_rows + 1):
        if (row, 0) in thicklines_table.get_celld():
            cell = thicklines_table[row, 0]
            cell.visible_edges += 'L'
            cell.set_linewidth(2.0)
        if (row, n_cols - 1) in thicklines_table.get_celld():
            cell = thicklines_table[row, n_cols - 1]
            cell.visible_edges += 'R'
            cell.set_linewidth(2.0)

    cell = thicklines_table[1, -1]
    cell.visible_edges += 'T'
    cell.set_linewidth(2.0)

    for row in range(len(cell_text) + 1):  # +1 for header
        if (row, -1) in colors_table.get_celld():
            colors_table[row, -1].get_text().set_ha('right')

    plt.savefig(
        fname=os.path.join(os.getcwd(), '..', 'output', 'figs', 'ffs_vs_best_feature_sets.png'),
        dpi=600,
        bbox_inches='tight',
        format='png'
    )


if __name__ == '__main__':
    main()
