basins = ['animas', 'crystal', 'jemez', 'logan', 'oak']


basin_titles = {
    'animas': 'Animas River',
    'crystal': 'Crystal River',
    'jemez': 'Jemez River',
    'logan': 'Logan River',
    'oak': 'Oak Creek'
}


snotel_stations = {
    'animas': [
        "Cascade", "Idarado", "Lizard Head Pass", "Lone Cone", "Mineral Creek", "Red Mountain Pass",
        "Slumgullion"
    ],
    'crystal': [
        "Brumley", "Butte", "Independence Pass", "Mc Clure Pass"
    ],
    'jemez': [
        "Quemazon", "Senorita Divide #2"
    ],
    'logan': [
        "Ben Lomond Peak", "Ben Lomond Trail", "Bug Lake", "Dry Bread Pond", "Franklin Basin", "Horse Ridge",
        "Little Bear", "Monte Cristo", "Tony Grove Lake"
    ],
    'oak': [
        "Fry", "Mormon Mountain", "White Horse Lake"
    ]
}


basin_files = {
        'animas': {
            'shapefile': 'data/shapefiles/Animas_River_09361500.shp',
            'streamflow': 'data/streamflow/Animas_River_09361500.csv'
        },
        'crystal': {
            'shapefile': 'data/shapefiles/Crystal_River_09081600.shp',
            'streamflow': 'data/streamflow/Crystal_River_09081600.csv'
        },
        'jemez': {
            'shapefile': 'data/shapefiles/Jemez_River_08324000.shp',
            'streamflow': 'data/streamflow/Jemez_River_08324000.csv'
        },
        'logan': {
            'shapefile': 'data/shapefiles/Logan_River_10109000.shp',
            'streamflow': 'data/streamflow/Logan_River_10109000.csv'
        },
        'oak': {
            'shapefile': 'data/shapefiles/Oak_Creek_09504500.shp',
            'streamflow': 'data/streamflow/Oak_Creek_09504500.csv'
        }
    }


basin_features = {
    'animas': [
        "SNOTEL_SWE_M_0", "SNOTEL_SWE_A_0", "SNOTEL_PA_M_0", "SNOTEL_PA_A_0", "SNOTEL_SWE_M_1", "SNOTEL_SWE_A_1",
        "SNOTEL_PA_M_1", "SNOTEL_PA_A_1", "SNOTEL_SWE_M_2", "SNOTEL_SWE_A_2", "SNOTEL_PA_M_2", "SNOTEL_PA_A_2",
        "SNOTEL_SWE_M_3", "SNOTEL_SWE_A_3", "SNOTEL_PA_M_3", "SNOTEL_PA_A_3", "SNOTEL_SWE_M_4", "SNOTEL_SWE_A_4",
        "SNOTEL_PA_M_4", "SNOTEL_PA_A_4", "SNOTEL_SWE_M_5", "SNOTEL_SWE_A_5", "SNOTEL_PA_M_5", "SNOTEL_PA_A_5",
        "SNOTEL_SWE_M_6", "SNOTEL_SWE_A_6", "SNOTEL_PA_M_6", "SNOTEL_PA_A_6", "SPFH_mean", "TMP_mean", "WIND_mean",
        "SOI", "PDO", "NAO", "AMO"
    ],
    'crystal': [
        "SNOTEL_SWE_M_0", "SNOTEL_SWE_A_0", "SNOTEL_PA_M_0", "SNOTEL_PA_A_0", "SNOTEL_SWE_M_1",  "SNOTEL_SWE_A_1",
        "SNOTEL_PA_M_1", "SNOTEL_PA_A_1", "SNOTEL_SWE_M_2", "SNOTEL_SWE_A_2", "SNOTEL_PA_M_2", "SNOTEL_PA_A_2",
        "SNOTEL_SWE_M_3", "SNOTEL_SWE_A_3", "SNOTEL_PA_M_3", "SNOTEL_PA_A_3", "SPFH_mean", "TMP_mean", "WIND_mean",
        "SOI", "PDO", "NAO", "AMO"
    ],
    'jemez': [
        "SNOTEL_SWE_M_0", "SNOTEL_SWE_A_0", "SNOTEL_PA_M_0", "SNOTEL_PA_A_0", "SNOTEL_SWE_M_1", "SNOTEL_SWE_A_1",
        "SNOTEL_PA_M_1", "SNOTEL_PA_A_1", "SPFH_mean", "TMP_mean", "WIND_mean", "SOI", "PDO", "NAO", "AMO"
    ],
    'logan': [
        "SNOTEL_SWE_M_0", "SNOTEL_SWE_A_0", "SNOTEL_PA_M_0", "SNOTEL_PA_A_0", "SNOTEL_SWE_M_1", "SNOTEL_SWE_A_1",
        "SNOTEL_PA_M_1", "SNOTEL_PA_A_1", "SNOTEL_SWE_M_2", "SNOTEL_SWE_A_2", "SNOTEL_PA_M_2", "SNOTEL_PA_A_2",
        "SNOTEL_SWE_M_3", "SNOTEL_SWE_A_3", "SNOTEL_PA_M_3", "SNOTEL_PA_A_3", "SNOTEL_SWE_M_4", "SNOTEL_SWE_A_4",
        "SNOTEL_PA_M_4", "SNOTEL_PA_A_4", "SNOTEL_SWE_M_5", "SNOTEL_SWE_A_5", "SNOTEL_PA_M_5", "SNOTEL_PA_A_5",
        "SNOTEL_SWE_M_6", "SNOTEL_SWE_A_6", "SNOTEL_PA_M_6", "SNOTEL_PA_A_6", "SNOTEL_SWE_M_7", "SNOTEL_SWE_A_7",
        "SNOTEL_PA_M_7", "SNOTEL_PA_A_7", "SNOTEL_SWE_M_8", "SNOTEL_SWE_A_8", "SNOTEL_PA_M_8", "SNOTEL_PA_A_8",
        "SPFH_mean", "TMP_mean", "WIND_mean", "SOI", "PDO", "NAO", "AMO"
    ],
    'oak': [
        "SNOTEL_SWE_M_0", "SNOTEL_SWE_A_0", "SNOTEL_PA_M_0", "SNOTEL_PA_A_0", "SNOTEL_SWE_M_1", "SNOTEL_SWE_A_1",
        "SNOTEL_PA_M_1", "SNOTEL_PA_A_1", "SNOTEL_SWE_M_2", "SNOTEL_SWE_A_2", "SNOTEL_PA_M_2", "SNOTEL_PA_A_2",
        "SPFH_mean", "TMP_mean", "WIND_mean", "SOI", "PDO", "NAO", "AMO"
    ]
}


# For Fixed vs. Best skill figure (Figure 6)
yticks = {
    'animas': [0, 1000, 2000, 3000],
    'crystal': [0, 450, 900, 1350],
    'jemez': [0, 150, 300, 450],
    'logan': [0, 350, 700, 1050],
    'oak': [0, 50, 100, 150]
}
