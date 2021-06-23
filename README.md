# Predicting Zillow Home Values


## Goal

The goal is to create a model using clustering methodologies to better predict the logerror in the current Zestimate on Zillow.

The log error is defined as logerror=log(Zestimate)−log(SalePrice)


### Data Dictionary

| Feature                      | Description                                                                                                        | Data Type                      |
|------------------------------|--------------------------------------------------------------------------------------------------------------------|--------------------------------|
| prop.parcelid                | Unique ID for properties assigned by the tax assessment office                                                     | discrete, integer              |
| pred.logerr                  | How far the predicted value is from the actual observation                                                         | continuous, float              |
| bathroomcnt                  | Number of bathrooms                                                                                                | discrete, integer, categorical |
| bedroomcnt                   | Number of bedrooms                                                                                                 | discrete, integer, categorical |
| calculatedfinishedsquarefeet | Square feet of living area                                                                                         | continuous, float              |
| fips                         | Federal Information Processing Standard code - see https://en.wikipedia.org/wiki/FIPS_county_code for more details | discrete, integer, categorical |
| latitude                     | Latitude of property location                                                                                      | continuous, float              |
| longtitude                   | Longitude of property location                                                                                     | continuous, float              |
| lotsizesquarefeet            | Size of the lot in square feet                                                                                     | continuous, float              |
| propertylandusetypeid        | ID of property land use type                                                                                       | discrete, integer              |
| regionidcity                 | ID of the city                                                                                                     | discrete, integer, categorical |
| regionidcounty               | ID of the county                                                                                                   | discrete, integer, categorical |
| regionidzip                  | Zip code                                                                                                           | discrete, integer, categorical |
| yearbuilt                    | Year built                                                                                                         | discrete, integer              |
| structuretaxvaluedollarcnt   | Assessed value of the built structure                                                                              | continuous, float              |
| taxvaluedollarcnt            | Total tax assessed value                                                                                           | continuous, float              |
| landtaxvaluedollarcnt        | Assessed tax value of the land                                                                                     | continuous, float              |
| taxamount                    | Total property tax assessed                                                                                        | continuous, float              |

## Project Steps

### Acquire

The Acquire.py module contains several funtions:
- Sets up the connection to the MySQL database where the data is stored.
- Acquires the Zillow data according to a SQL statement.
- Caches the acquired data locally to speed up processing.

### Prepare

### Explore

### Model

### Conclusion
