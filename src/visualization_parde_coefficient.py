import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.configs.basins import basins, basin_titles
from src.utils.data_loaders import get_streamflow_data

matplotlib.use('Qt5Agg')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'Times New Roman',
    'axes.titlesize': 18.5,
    'axes.labelsize': 18,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17
})


def main():
    labels = ['Oct.', 'Nov.', 'Dec.', 'Jan.', 'Feb.', 'Mar.', 'Apr.', 'May', 'Jun.', 'Jul.', 'Aug.', 'Sep.']

    for basin in basins:

        years = np.arange(1981, 2021) if basin != 'jemez' else np.arange(1981, 2020)

        data = get_streamflow_data(basin)
        data = data[data['year'].isin(years)].drop(['year', 'avg_AMJJ'], axis=1)
        data['total'] = data.sum(axis=1)

        avg_monthly, avg_annual = data.mean()[:12].tolist(), data.mean()[-1].tolist()
        parde_coefficient = [x / avg_annual for x in avg_monthly]

        parde_wy = parde_coefficient[9:] + parde_coefficient[:9]

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

        ax.plot(parde_wy, marker='o', zorder=2)
        ax.grid(True, axis='x', linestyle='--', alpha=0.4)
        ax.set_xticks(np.arange(0, 12), labels=labels)
        ax.set_ylabel('Pardé Coefficient')
        ax.set_xlim(-0.25, 11.25)
        ax.set_title(f'Runoff Seasonality - {basin_titles[basin]}')
        ylims = ax.get_ylim()
        ax.axvline(x=6, color='red', linestyle='--', zorder=1)
        ax.axvline(x=9, color='red', linestyle='--', zorder=1)

        fig.tight_layout()

        plt.savefig(
            fname=os.path.join(os.getcwd(), '..', 'output', 'figs', f'{basin}_parde_coefficient.png'),
            dpi=600,
            bbox_inches='tight',
            format='png'
        )


if __name__ == '__main__':
    main()
