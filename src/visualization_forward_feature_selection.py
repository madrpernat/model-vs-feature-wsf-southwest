import os
from string import capwords

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.configs.basins import basins
from src.configs.features import acronyms, colors
from src.configs.regressors import regressors
from src.utils.data_loaders import  get_ffs_order, get_ffs_iteration_scores

matplotlib.use('Qt5Agg')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'Times New Roman',
    'axes.titlesize': 17,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})


def main():

    n_basins = len(basins)
    metric = 'rrmse'

    for regressor in regressors:

        fig, axes = plt.subplots(nrows=n_basins, ncols=1, figsize=(7.7, 12.5))
        plt.subplots_adjust(
            top=0.965,
            bottom=0.085,
            left=0.160,
            right=0.95,
            hspace=1.15
        )

        for i, basin in enumerate(basins):

            ffs_order = get_ffs_order(basin)
            ffs_skill_gain = get_ffs_iteration_scores(basin, regressor)

            selected_features = ffs_order[ffs_order[regressor] > 0].sort_values(by=regressor).index.tolist()[:10]
            rrmse_values = ffs_skill_gain[metric].to_list()[:10]

            # X-axis positions (numeric values for plotting)
            x_positions = np.arange(1, len(selected_features) + 1)

            ax = axes[i]
            ax.plot(x_positions, rrmse_values, marker='o', linestyle='-', color='black', markersize=8, label='RRMSE')
            ax.set_title(capwords(basin))
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.set_ylabel('RRMSE', fontsize=17)

            ax.set_xticks(x_positions)
            ax.set_xticklabels([])

            ymin, ymax = ax.get_ylim()
            y_position = ymin - 0.11 * (ymax - ymin)

            for x_position, selected_feature in zip(x_positions, selected_features):

                feature_name = acronyms[selected_feature]
                color = colors[selected_feature]

                ax.text(x_position, y_position, feature_name, fontsize=14, ha='center', va='top', rotation=90,
                        bbox=dict(facecolor=color, edgecolor='black', boxstyle='round,pad=0.3'))

        plt.savefig(
            fname=os.path.join(os.getcwd(), '..', 'output', 'figs', f'{regressor}_ffs.png'),
            dpi=600,
            bbox_inches='tight',
            format='png'
        )


if __name__ == '__main__':
    main()
