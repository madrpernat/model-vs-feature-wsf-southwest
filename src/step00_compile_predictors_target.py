from datetime import date
import os
from typing import List

import netCDF4
import numpy as np
from numpy import ndarray
import osgeo
import osgeo.gdal
import osgeo.ogr
import pandas as pd

from src.configs.basins import basins, basin_files, snotel_stations
from src.configs.nldas_vars import nldas_vars


def make_mask(
        lons: ndarray,
        lats: ndarray,
        cell_size: float,
        shapefile: str
) -> ndarray:

    # Open the source shapefile
    source_ds = osgeo.ogr.Open(shapefile)
    source_layer = source_ds.GetLayer()

    # Create a high-resolution raster in memory
    mem_ds = osgeo.gdal.GetDriverByName('MEM').Create('', lons.size, lats.size, osgeo.gdal.GDT_Byte)
    mem_ds.SetGeoTransform((lons.min(), cell_size, 0, lats.min(), 0, cell_size))
    band = mem_ds.GetRasterBand(1)

    # Rasterize the shapefile to the grid
    osgeo.gdal.RasterizeLayer(mem_ds, [1], source_layer, burn_values=[1])

    # Convert the rasterized data to a numpy array and clean up
    array = band.ReadAsArray()
    mem_ds = None
    band = None

    return array


def find_var_in_month(
        yearmonth_idx: int,
        var: str,
        mask: ndarray,
        nldas_files: List[str]
) -> ndarray:

    # Open the NLDAS dataset file for the given year and month
    nldas_data = netCDF4.Dataset(nldas_files[yearmonth_idx])

    # Select the variable data based on its dimensions
    if len(nldas_data.variables[var].shape) == 3:
        var_data_for_mask = nldas_data.variables[var][0, :, :]
    else:
        var_data_for_mask = nldas_data.variables[var][0, 0, :, :]

    # Apply the mask to the selected variable data
    nldas_1basin_1var = np.ma.masked_where(mask == 0, var_data_for_mask)

    return nldas_1basin_1var


def main():

    # Build nldas filenames which are indexed by year-month
    start_year = 1981
    end_year = 2021
    start_dt = date(start_year - 1, 10, 1)  # Minus 1 year to account for WY months in previous CY
    end_dt = date(end_year, 1, 1)

    # Generate list of year-month strings
    yearmonth = [dt.strftime("%Y%m") for dt in pd.date_range(start_dt, end_dt, freq='M')]

    nldas_filenames = [
        os.path.join(os.getcwd(), 'data', 'nldas', f'NLDAS_FORA0125_M.A{ym}.002.grb.nc')
        for ym in yearmonth
    ]

    # Get lat, lon, and cell_size information for NLDAS gridded data
    nldas_data_1month = netCDF4.Dataset(nldas_filenames[0], 'r')
    lats = nldas_data_1month.variables["lat"][:]
    lons = nldas_data_1month.variables["lon"][:]
    cell_size = lons[:][1] - lons[:][0]

    # Initialize a dictionary to store all features and target variable (AMJJ water supply) for all basins for each year
    data_dict = {
        'Year': [],
        'Basin': [],
        'Streamflow': [],
        'AMO': [],
        'SOI': [],
        'NAO': [],
        'PDO': []
    }

    # To the data_dict, add SNOTEL station features
    max_snotel_sites = max([len(snotel_stations[key]) for key in snotel_stations])
    for i in range(max_snotel_sites):
        data_dict.update({
            f'SNOTEL_SWE_M_{i}': [],
            f'SNOTEL_SWE_A_{i}': [],
            f'SNOTEL_PA_M_{i}': [],
            f'SNOTEL_PA_A_{i}': [],
        })

    # To data_dict, add NLDAS features
    for var_name in nldas_vars.values():
        data_dict.update({
            f'{var_name}_mean': []
        })

    # Load climate indices
    amo = pd.read_csv('data/climate_features/amo_yearly.csv').set_index('year')
    soi = pd.read_csv('data/climate_features/soi_yearly.csv').set_index('year')
    pdo = pd.read_csv('data/climate_features/pdo_yearly.csv').set_index('year')
    nao = pd.read_csv('data/climate_features/nao_yearly.csv').set_index('year')

    # Load snotel data
    snotel_data = pd.read_csv('data/snotel/snotel_data_all_stations.csv')
    snotel_data = snotel_data.set_index('Station Name')

    for basin in basins:

        flow_data = pd.read_csv(basin_files[basin]['streamflow']).set_index('year')

        shapefile = basin_files[basin]['shapefile']
        mask = make_mask(lons, lats, cell_size, shapefile)

        for year in range(start_year, 2021):

            data_dict['Year'].append(year)
            data_dict['Basin'].append(basin)

            data_dict['Streamflow'].append(flow_data.loc[year, 'avg_AMJJ'])

            data_dict['AMO'].append(amo.loc[year][0])
            data_dict['SOI'].append(soi.loc[year][0])
            data_dict['PDO'].append(pdo.loc[year][0])
            data_dict['NAO'].append(nao.loc[year][0])

            for snotel_num in range(max_snotel_sites):

                if snotel_num >= len(snotel_stations[basin]):
                    # Append zero value for all features if snotel number is out of range for current basin
                    for feature in ['SWE_M', 'SWE_A', 'PA_M', 'PA_A']:
                        data_dict[f'SNOTEL_{feature}_{snotel_num}'].append(0)

                else:
                    # Access data for current snotel site
                    tmp_df = snotel_data.loc[snotel_stations[basin][snotel_num]].set_index('Water Year')
                    tmp_year = tmp_df.loc[year]

                    feature_columns = {
                        'SWE_M': 10,
                        'SWE_A': 12,
                        'PA_M': 11,
                        'PA_A': 13
                    }

                    for feature, feature_column in feature_columns.items():
                        data_dict[f'SNOTEL_{feature}_{snotel_num}'].append(tmp_year.iloc[feature_column])

            # Populate nldas data
            for var_id, var_name in nldas_vars.items():
                # Calculate spatial mean for each month from October to March
                monthly_var_values = [
                    np.mean(find_var_in_month(
                        yearmonth_idx=12 * (year - start_year) + month_offset,
                        var=var_id,
                        mask=mask,
                        nldas_files=nldas_filenames
                    ))
                    for month_offset in range(6)
                ]
                # Aggregate the monthly values into a yearly value
                longTerm_mean = np.mean(monthly_var_values)

                data_dict[f'{var_name}_mean'].append(longTerm_mean)

    # Create new column that is the magnitude of the zonal and meridional wind components
    u = np.array(data_dict['UGRD_mean'])
    v = np.array(data_dict['VGRD_mean'])

    data_dict['WIND_mean'] = np.linalg.norm(np.stack((u, v), axis=-1), axis=1).tolist()

    # Convert dictionary into a dataframe
    data_df = pd.DataFrame(data_dict)

    # Save to .csv
    data_df.to_csv('data/full_feature_target_data.csv', index=False)


if __name__ == '__main__':
    main()
