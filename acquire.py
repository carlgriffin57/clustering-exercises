import numpy as np
import pandas as pd
import os

# Enables access to my env.py file in order to use sensitive info to access Codeup DB

from env import host, username, password

# sets up a secure connection to the Codeup db using my login infor
def get_db_url(db, user=username, host=host, password=password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

# assigns the zillow url to the variable name 'url' so it can be used in additional functions
url = get_db_url('zillow')

def get_connection(db, user=username, host=host, password=password):
    '''
    This function uses my env file to create a connection url to access
    the Codeup database. '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def new_zillow_data():
    '''
    This function reads data from the Codeup db into a df.
    '''

    sql_query = '''SELECT *
                FROM properties_2017 
                JOIN predictions_2017 using (parcelid) 
                left JOIN propertylandusetype using (propertylandusetypeid) 
	            left join airconditioningtype using (airconditioningtypeid)
	            left join architecturalstyletype using (architecturalstyletypeid)
                left join buildingclasstype using (buildingclasstypeid)
               	left join heatingorsystemtype using (heatingorsystemtypeid)
	            left join storytype using (storytypeid)
             	left join typeconstructiontype using (typeconstructiontypeid) 
                WHERE month(transactiondate) >= 01 and month(transactiondate) <= 12 
 				and latitude > 0 and longitude < 0;'''
    return pd.read_sql(sql_query, get_connection('zillow'))


def get_zillow_data():
    '''
    This function reads in Zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow_df.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_zillow_data()
     
        # Cache data
        df.to_csv('zillow_df.csv')
        
    return df