import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Function to read messages and categories file and merge into single dataframe df
    based on 'id' column
    
    Arguments:
        messages_filepath -> Path to the CSV file containing messages
        categories_filepath -> Path to the CSV file containing categories
    Output:
        df -> Combined dataframe containing messages and categories
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id', how='inner')
    return df 


def clean_data(df):
    """
    Clean Categories Data set to make it compatible to be used in models for machine learning algorithms
    
    Arguments:
        df -> Combined dataframe containing messages and categories
    Outputs:
        df -> Combined dataframe containing messages and categories with categories cleaned up
    """
    # Split the categories column with delimit ';'   
    categories = df['categories'].str.split(pat=';',expand=True)
    
    #Convert the first row values in categories dataframe to the column headers. 

    row = categories.iloc[[1]]
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(np.int)
    
    # Drop the duplicate rows from df dataframe
    #Remove the existing categories column from the df dataframe and concat the formatted 
    #categories dataframe with df dataframe.   
    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
     Save Data to SQLite Database Function
    
     Arguments:
        df -> Cleaned dataframe containing Categories and messagages data
        database_filename -> Path to SQLite destination database
    """  
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('msgcategory', engine, index=False, if_exists='replace') 


def main():
    """
    Main function which will kick off the data processing functions. There are three primary actions taken by this function:
        1) Load Messages Data with Categories
        2) Clean Categories Data
        3) Save Data to SQLite Database
    """
    # Print the system arguments
    print(sys.argv)
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()