# cityLocs
## Author: Cutter Beck

This directory explores the city dataset I collected.

### Files
- `uscities.csv`: the dataset from https://simplemaps.com/data/us-cities which I use to plot and group US cities
- `worldcities.csv`: unused, contains cities from around the world
- `gsv_image_locations.csv`: contains an extraction of all images from the USA GSV dataset I curated. Columns are:
    - `image`: the image file name
    - `city`: the city the image is from
    - `lat`: the latitude of the image
    - `lng`: the longitude of the image
- `kmeansCluster.py`
    - Clusters cities into k clusters and displays them
- `locationDocCreator.py`
    - reads the image directory and creates a csv of the image locations and GPS coordinates
- `processCities.py`
    - A file I play with to see how best to visualize and analyze the city data
    - Currently showcases a Voronoi diagram of cities in the "north east"