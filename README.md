## Overview

This repository contains the code and data used in the study:

**"The Relative Importance of Model Structure and Input Features for Water Supply Forecasting in Snow-Dominated River Basins of the Southwest US"**  
(*Submitted to* *The Journal of Hydrology: Regional Studies*)

This study explores how model type and feature selection influence April–July (AMJJ) water supply forecast (WSF) skill in five snow-influenced watersheds in the southwestern United States. We compare five machine learning model types and introduce a new wrapper-based feature selection method—**Semi-Exhaustive Feature Search (SEFS)**—to identify the most predictive features for each basin–model type combination.

---

## Data Sources

The `data/` directory contains all of the datasets used in this study, organized by category. All data are already preprocessed and ready for use—users do not need to download from external sources unless desired.

- **Streamflow** (`data/streamflow/`): Monthly streamflow volumes for each basin were obtained from the [USGS National Water Information System (NWIS)](https://waterdata.usgs.gov/nwis/sw).
- **SNOTEL** (`data/snotel/`): Snow water equivalent (SWE) and precipitation accumulation (PA) were gathered from the [NRCS SNOTEL network](https://nwcc-apps.sc.egov.usda.gov/?networkFilters=sntl).
- **Meteorological Variables** (`data/nldas/`): Monthly averages of temperature, specific humidity, and wind speed were derived from the [NLDAS-2 dataset](https://disc.gsfc.nasa.gov/datasets/NLDAS_FORA0125_M_2.0/summary?keywords=NLDAS).
- **Climatological Indices** (`data/climate_features/`):
  - Southern Oscillation Index (SOI): [NOAA CPC](https://www.cpc.ncep.noaa.gov/data/indices/)
  - Pacific Decadal Oscillation (PDO): [NOAA NCEI](https://www.ncei.noaa.gov/access/monitoring/pdo/)
  - Atlantic Multidecadal Oscillation (AMO): [NOAA PSL](https://psl.noaa.gov/data/timeseries/AMO/)
  - North Atlantic Oscillation (NAO): [UCAR Climate Data Guide](https://climatedataguide.ucar.edu/climate-data/hurrell-north-atlantic-oscillation-nao-index-pc-based)
- **Shapefiles** (`data/shapefiles/`): Watershed boundaries were downloaded using the [USGS StreamStats tool](https://streamstats.usgs.gov/ss/).

---

## Repository Structure

*To be added.*

---

## How to Use

*To be added.*