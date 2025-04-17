import os

import matplotlib.pyplot as plt
import matplotlib

from src.utils.utils import timeseries_plot
from src.configs.basin_names import basins
from src.configs.regressors import regressors
from src.configs.basin_yticks import yticks

matplotlib.use('Qt5Agg')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'Times New Roman',
    'axes.titlesize': 19,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})


def main():

    for regressor in regressors:

        fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(8.5, 10))
        plt.subplots_adjust(
            top=0.955,
            bottom=0.100,
            left=0.135,
            right=0.95,
            hspace=0.405,
            wspace=0.2
        )

        for i, basin in enumerate(basins):

            timeseries_plot(
                ax=axes[i],
                basin=basin,
                regressor=regressor,
                yticks=yticks[basin]
            )

            # Enable vertical grid lines for all plots
            axes[i].grid(True, axis='x', linestyle='--', alpha=0.4)

            # Only bottom plot gets x-axis ticks and labels
            if i != len(axes) - 1:
                axes[i].set_xlabel('')
                axes[i].tick_params(axis='x', labelbottom=False, bottom=False)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles=handles,
            labels=labels,
            loc='lower center',
            fontsize=14,
            ncol=3,
            handletextpad=0.1,
            markerscale=1.3,
            borderpad=0.3,
            columnspacing=1
        )

        plt.savefig(
            fname=os.path.join(os.getcwd(), '..', 'output', 'figs', f'{regressor}__best_vs_swe_preds.png'),
            dpi=600,
            bbox_inches='tight',
            format='png'
        )


if __name__ == '__main__':
    main()
