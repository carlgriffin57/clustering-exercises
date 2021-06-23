import pandas as pd
import numpy as np
import os
from env import host, username, password
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ignore warnings
import warnings
warnings.filterwarnings("ignore")


# Statistical Tests
import scipy.stats as stats



#Summarize Data 

def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # create a mask of columns whether they are object type or not
    mask = np.array(df.dtypes == "object")

        
    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()
    
    return object_cols

def get_numeric_cols(df, object_cols):
    '''
    takes in a dataframe and list of object column names
    and returns a list of all other columns names, the non-objects. 
    '''
    numeric_cols = [col for col in df.columns.values if col not in object_cols]
    
    return numeric_cols

# def get_single_use_prop(df):
#     single_use = [261, 262, 263, 264, 266, 268, 273, 276, 279]
#     df = df[df.propertylandusetypeid.isin(single_use)]
#     return df 

def handle_missing_values(df, prop_required_row = 0.5, prop_required_col = 0.5):
    ''' funtcion which takes in a dataframe, required notnull proportions of non-null rows and columns.
    drop the columns and rows columns based on theshold:'''
    
    #drop columns with nulls
    threshold = int(prop_required_col * len(df.index)) # Require that many non-NA values.
    df.dropna(axis = 1, thresh = threshold, inplace = True)
    
    #drop rows with nulls
    threshold = int(prop_required_row * len(df.columns)) # Require that many non-NA values.
    df.dropna(axis = 0, thresh = threshold, inplace = True)
    
    
    return df

def get_latitude(df):
    '''
    This function takes in a datafame with latitude formatted as a float,
    converts it to a int and utilizes lambda to return the latitude values
    in a correct format.
    '''
    df.latitude = df.latitude.astype(int)
    df['latitude'] = df['latitude'].apply(lambda x: x / 10 ** (len((str(x))) - 2))
    return df

def get_longitude(df):
    '''This function takes in a datafame with longitude formatted as a float,
    converts it to a int and utilizes lambda to return the longitude values
    in the correct format.
    '''
    df.longitude = df.longitude.astype(int)
    df['longitude'] = df['longitude'].apply(lambda x: x / 10 ** (len((str(x))) - 4))
    return df

def clean_zillow(df):
    df = get_single_use_prop(df)

    df = handle_missing_values(df, prop_required_row = 0.5, prop_required_col = 0.5)

    df.set_index('parcelid', inplace=True)

    # cols_to_drop = ['fullbathcnt','heatingorsystemtypeid','finishedsquarefeet12', 
    #             'propertycountylandusecode', 'propertylandusetypeid','propertyzoningdesc', 'censustractandblock',
    #             'propertylandusedesc', 'buildingqualitytypeid' , 'unitcnt', 'heatingorsystemdesc', 
    #             'lotsizesquarefeet','regionidcity', 'calculatedbathnbr', 'transactiondate', 'roomcnt', 'id', 'regionidcounty',
    #             'regionidzip', 'assessmentyear']

    # df.drop(columns=cols_to_drop, inplace = True)

    df.dropna(inplace = True)

    get_latitude(df)

    get_longitude(df)

    return df

#def get_county(df):
    #Convert fips to int
    # df.fips = df.fips.astype('int64')

    # county = []

    # for row in df['fips']:
    #     if row == 6037:
    #         county.append('Los Angeles')
    #     elif row == 6059:
    #         county.append('Orange')
    #     elif row == 6111:
    #         county.append('Ventura')
        
    # df['county'] = county

    # df.drop(columns={'fips'}, inplace=True)
    # return df

def get_counties(df):
    '''
    This function will create dummy variables out of the original fips column. 
    And return a dataframe with all of the original columns except regionidcounty.
    We will keep fips column for data validation after making changes. 
    New columns added will be 'LA', 'Orange', and 'Ventura' which are boolean 
    The fips ids are renamed to be the name of the county each represents. 
    '''
    # create dummy vars of fips id
    county_df = pd.get_dummies(df.fips)
    # rename columns by actual county name
    county_df.columns = ['LA', 'Orange', 'Ventura']
    # concatenate the dataframe with the 3 county columns to the original dataframe
    df_dummies = pd.concat([df, county_df], axis = 1)
    # drop regionidcounty and fips columns
    # df_dummies = df_dummies.drop(columns = ['regionidcounty'])
    return df_dummies

def create_dummies(df, object_cols):
    '''
    This function takes in a dataframe and list of object column names,
    and creates dummy variables of each of those columns. 
    It then appends the dummy variables to the original dataframe. 
    It returns the original df with the appended dummy variables. 
    '''
    
    # run pd.get_dummies() to create dummy vars for the object columns. 
    # we will drop the column representing the first unique value of each variable
    # we will opt to not create na columns for each variable with missing values 
    # (all missing values have been removed.)
    dummy_df = pd.get_dummies(df[object_cols], dummy_na=False, drop_first=True)
    
    # concatenate the dataframe with dummies to our original dataframe
    # via column (axis=1)
    df = pd.concat([df, dummy_df], axis=1)

    return df
def remove_outliers():
    '''
    remove outliers in bed, bath, zip, square feet, acres & tax rate
    '''

    return df[((df.bathroomcnt <= 7) & (df.bedroomcnt <= 7) & 
               (df.regionidzip < 100000) & 
               (df.bathroomcnt > 0) & 
               (df.bedroomcnt > 0) & 
               (df.acres < 20) &
               (df.calculatedfinishedsquarefeet < 10000) & 
               (df.taxrate < 10)
              )]

def split(df, target_var):
    '''
    This function takes in the dataframe and target variable name as arguments and then
    splits the dataframe into train (56%), validate (24%), & test (20%)
    It will return a list containing the following dataframes: train (for exploration), 
    X_train, X_validate, X_test, y_train, y_validate, y_test
    '''
    # split df into train_validate (80%) and test (20%)
    train_validate, test = train_test_split(df, test_size=.20, random_state=13)
    # split train_validate into train(70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=13)

    # create X_train by dropping the target variable 
    X_train = train.drop(columns=[target_var])
    # create y_train by keeping only the target variable.
    y_train = train[[target_var]]

    # create X_validate by dropping the target variable 
    X_validate = validate.drop(columns=[target_var])
    # create y_validate by keeping only the target variable.
    y_validate = validate[[target_var]]

    # create X_test by dropping the target variable 
    X_test = test.drop(columns=[target_var])
    # create y_test by keeping only the target variable.
    y_test = test[[target_var]]

    partitions = [train, X_train, X_validate, X_test, y_train, y_validate, y_test]
    return partitions

def scale_my_data(train, validate, test):
    #call numeric cols
    numeric_cols = get_numeric_cols(df)
    scaler = StandardScaler()
    scaler.fit(train[[numeric_cols]])

    X_train_scaled = scaler.transform(train[[numeric_cols]])
    X_validate_scaled = scaler.transform(validate[[numeric_cols]])
    X_test_scaled = scaler.transform(test[[numeric_cols]])

    train[[numeric_cols]] = X_train_scaled
    validate[[numeric_cols]] = X_validate_scaled
    test[[numeric_cols]] = X_test_scaled
    return train, validate, test

def prepare_zillow(df):
    #Separate logerror into quantiles
    df['logerror_class'] = pd.qcut(df.logerror, q=4, labels=['q1', 'q2', 'q3', 'q4'])
    df = get_counties(df)
    return df

    #Split data into Train, Validate, and Test
    #train, validate, test = train_validate_test_split(df, target='logerror_class', seed=123)
    #train, validate, test = scale_my_data(train, validate, test)

    #return train, validate, test