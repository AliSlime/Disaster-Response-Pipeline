import sys
import pandas as pd
import re
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    load origin files
    :param messages_filepath: message file path
    :param categories_filepath: category file path
    :return: df
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    ## merge table
    df = messages.merge(categories, how='left', on=['id'])

    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(';', expand=True)

    row = categories.loc[0]

    category_colnames = [re.sub('-.*$', '', x) for x in row]

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.replace('^.*-', '')

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.merge(df, categories, left_on=df.index.values, right_on=categories.index.values)
    df.drop(['key_0'], axis=1, inplace=True)

    return df


def clean_data(df):
    """
    clean dataframe's data
    :param df: dataframe
    :return: result
    """
    return df.drop_duplicates()


def save_data(df, database_filename):
    """
    save data from dataframe
    :param df: dataframe
    :param database_filename: open sqlite database file name
    :return:
    """
    # print('sqlite:///{}'.format(database_filename))
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('InsertTableName', engine, index=False)


def main():
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