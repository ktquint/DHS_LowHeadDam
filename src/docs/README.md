# DHS_LowHeadDam Repository

This code allows the user to analyze the downstream flow conditions at a low-head dam 
if provided with the dam location (latitude, longitude), and dam width (m). 

*This code is meant to be used in tandem with the
[RathCelon](https://github.com/jlgutenson/rathcelon) and
[ARC](https://github.com/MikeFHS/automated-rating-curve) python packages.

## Data Acquisition
- Digital Elevation Models are downloaded from the USGS TNM api with 1-10 m resolution.
- NHDPlus High Resolution flowlines are downloaded from the USGS TNM api
- Streamflow estimation 
