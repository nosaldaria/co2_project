import pandas as pd


# Fetch the data
share_g_forest = pd.read_csv("https://ourworldindata.org/grapher/share-global-forest.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
share_land_forest = pd.read_csv("https://ourworldindata.org/grapher/forest-area-as-share-of-land-area.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
share_g_forest_area = pd.read_csv("https://ourworldindata.org/grapher/share-global-forest.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})


annual_change_forest_area = pd.read_csv("https://ourworldindata.org/grapher/annual-change-forest-area.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
annual_deforestation = pd.read_csv("https://ourworldindata.org/grapher/annual-deforestation.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})


share_g_forest.to_csv('../../datasets/forest/share_global_forest.csv', index=True)
share_land_forest.to_csv('../../datasets/forest/share_land_forest.csv', index=True)
share_g_forest_area.to_csv('../../datasets/forest/share_g_forest_area.csv', index=True)

annual_change_forest_area.to_csv('../../datasets/forest/annual_change_forest_area.csv', index=True)
annual_deforestation.to_csv('../../datasets/forest/annual_deforestation.csv', index=True)
