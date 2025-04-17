import os
from string import capwords

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.configs.basin_names import basins
from src.configs.regressors import regressors
from src.utils.utils import get_best_and_fixed_scores, lolipop_plot

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

    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(9, 12))
    plt.subplots_adjust(
        top=0.964,
        bottom=0.078,
        left=0.088,
        right=0.983,
        hspace=0.285,
        wspace=0.242
    )

    for i, basin in enumerate(basins):

        best_rrmse, best_nse, swe_rrmse, swe_nse = get_best_and_fixed_scores(regressors, basin)

        # Max ylim for RRMSE and NSE plots
        rrmse_lim = 0.65
        nse_lim = 1.18

        # y ticks
        rrmse_ticks = [x / 100 for x in range(0, int(rrmse_lim * 100), 10)]
        nse_ticks = [x / 10 for x in range(0, int(nse_lim * 10), 2)]

        # Make plots
        lolipop_plot(
            ax=axes[i, 0],
            regressors=regressors,
            data_a=best_rrmse,
            data_b=swe_rrmse,
            color_a='red',
            color_b='blue',
            label_a='Best Feature Set',
            label_b='Fixed Feature Set (SWE A)',
            ylims=(0, rrmse_lim),
            yticks=rrmse_ticks
        )
        axes[i, 0].set_ylabel(capwords(basin), fontsize=19, fontweight='bold', labelpad=15)

        lolipop_plot(
            ax=axes[i, 1],
            regressors=regressors,
            data_a=best_nse,
            data_b=swe_nse,
            color_a='red',
            color_b='blue',
            label_a='Best Feature Set',
            label_b='Fixed Feature Set (SWE A)',
            ylims=(0, nse_lim),
            yticks=nse_ticks,
            invert_y=True
        )

    axes[0, 0].set_title('RRMSE', fontweight='bold', fontsize=19)
    axes[0, 1].set_title('NSE', fontweight='bold', fontsize=19)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles=handles,
        labels=labels,
        loc='lower center',
        fontsize=16,
        ncol=2,
        markerscale=1.3,
        borderpad=0.3,
        handletextpad=-0.2,
        columnspacing=1
    )

    plt.savefig(
        fname=os.path.join(os.getcwd(), '..', 'output', 'figs', 'best_vs_swe_skill.png'),
        dpi=600,
        bbox_inches='tight',
        format='png'
    )


if __name__ == '__main__':
    main()
