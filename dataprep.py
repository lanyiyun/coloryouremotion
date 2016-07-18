
'''''''''''''''''''''''''''''''''
  Module for data preprocessing
'''''''''''''''''''''''''''''''''
import json
import pandas as pd
import pickle
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2

class Users:

    def __init__(self, school):
        '''
        Args:
             school - school name, e.g. 'Stanbridges', 'Brooklyn', 'NewEn', etc (string)
        '''
        self.df = pd.DataFrame()
        self.school = school
        self.file_location = '/Users/lanyiyun/Documents/Insight Project/Data/Consolidated_' + school + '.xlsx'

        #Define a database name
        self.dbname = 'user_db'
        self.username = 'lanyiyun'
        self.pswd = 'lanyiyun'

        #a connection to a database
        self.engine = create_engine('postgresql://%s:%s@localhost/%s'%(self.username,self.pswd,self.dbname))

    def _read_data_(self):
        self.df = pd.read_excel(self.file_location)

    def _clean_df_(self, remove_NULL=False, how='any', fill_NULL=False, fill_value=0):
        if remove_NULL:
            self.df = self.df.dropna(0, how)
            if fill_NULL:
                self.df = self.df.fillna(fill_value, 0)

    def dataprep(self, remove_NULL=False, how='any', fill_NULL=False, fill_value=0):
        '''
        data preprocessing
        Args: (optional)
             remove_NULL: False, not remove missing value; True, remove all missing values
             how: - approach to remove missing values, 'any' or 'all'
             fill_NULL: False, not fill missing data; True, fill missing data
             fill_value: value used to fill the missing data
        '''
        if not remove_NULL:
            self._read_data_()
        self._clean_df_(remove_NULL, how, fill_NULL, fill_value)

    # list the number of missing data for each event
    def describe_null(self):
        notnull = [len(self.df)-i for i in self.df.count(0)]
        notnull_ptg = [1.0*(len(self.df)-i)/len(self.df) for i in self.df.count(0)]
        #self.desnull_df =pd.DataFrame(notnull_ptg, columns = self.df.columns)
        missing_df = pd.DataFrame({'total missing values': notnull,
                                   'missing percentage': notnull_ptg,
                                   'data type': self.df.dtypes},
                                    index = self.df.columns.tolist())
        return missing_df

    # load pickle data
    def load(self):
        '''
        Loads existing cleaned data from a pickled dataframe
        '''
        con = None
        con = psycopg2.connect(database = self.dbname, user = self.username, host='localhost', password=self.pswd)

        # query:
        sql_query = """
                    SELECT * FROM user_table;
                    """
        user_data_from_sql = pd.read_sql_query(sql_query,con)

        return user_data_from_sql


    # save data to pickle
    def save(self, create_or_append):
        '''
        save data to database and to DataFrame pickle. User can choose
        whether to append to or overwrite the database table, but the
        DataFrame pickle will always be overwritten

        Args:
            create_or_append: 'create' or 'append'. Applies only to database table
            DataFrame pickle will always be overwritten (string)
        '''

        # Pickle DataFrame as extra backup
        pickle.dump(self.df, open('user_df.pickle', "w"))

        # Write to database
        ## create a database (if it doesn't exist)
        if not database_exists(self.engine.url):
            create_database(self.engine.url)

        self.df.to_sql('user_table', self.engine, if_exists='replace')


class Events:

    def __init__(self, school):
        '''
        Args:
             school - school name, e.g. 'Stanbridges', 'Brooklyn', 'NewEn', etc (string)
        '''
        self.df = pd.DataFrame(columns=['user_id',
                                        'browser',
                                        'city',
                                        'country',
                                        'device_type',
                                        'domain',
                                        'landing_page',
                                        'event_name',
                                        'time',
                                        'session_time',
                                        'path',
                                        'platform'])
        self.school = school
        self.file_location = '/Users/lanyiyun/Documents/Insight Project/Data/' + school + '/events.json'

        #Define a database name
        self.dbname = 'emote_db'
        self.username = 'lanyiyun'
        self.pswd = 'lanyiyun'

        #a connection to a database
        self.engine = create_engine('postgresql://%s:%s@localhost/%s'%(self.username,self.pswd,self.dbname))

    # read JSON data
    def _read_JSON_(self):
        data_events = []
        with open(self.file_location) as f:
            for line in f:
                data_events.append(json.loads(line))
        data_events = pd.read_json(json.dumps(data_events))
        return data_events

    # convert JSON data into dataframe
    def _extract_data_(self):

        df_to_be_extracted = self._read_JSON_()

        try:
            self.df['user_id'] =  [df_to_be_extracted['data'][i]['user_id'] if 'user_id' in
                                   df_to_be_extracted['data'][i] else np.nan for i in range(len(df_to_be_extracted))]
            self.df['browser'] =  [df_to_be_extracted['data'][i]['browser'] if 'browser' in
                                   df_to_be_extracted['data'][i] else np.nan for i in range(len(df_to_be_extracted))]
            self.df['city'] =  [df_to_be_extracted['data'][i]['city'] if 'city' in
                                df_to_be_extracted['data'][i] else np.nan for i in range(len(df_to_be_extracted))]
            self.df['country'] =  [df_to_be_extracted['data'][i]['country'] if 'country' in
                                   df_to_be_extracted['data'][i] else np.nan for i in range(len(df_to_be_extracted))]
            self.df['device_type'] =  [df_to_be_extracted['data'][i]['device_type'] if 'device_type' in
                                       df_to_be_extracted['data'][i] else np.nan for i in range(len(df_to_be_extracted))]
            self.df['domain'] =  [df_to_be_extracted['data'][i]['domain'] if 'domain' in
                                  df_to_be_extracted['data'][i] else np.nan for i in range(len(df_to_be_extracted))]
            self.df['landing_page'] =  [df_to_be_extracted['data'][i]['landing_page'] if 'landing_page' in
                                        df_to_be_extracted['data'][i] else np.nan for i in range(len(df_to_be_extracted))]
            self.df['event_name'] =  [df_to_be_extracted['name'][i] for i in range(len(df_to_be_extracted))]
            self.df['time'] =  [df_to_be_extracted['data'][i]['time'] if 'time' in
                                df_to_be_extracted['data'][i] else np.nan for i in range(len(df_to_be_extracted))]
            self.df['session_time'] =  [df_to_be_extracted['data'][i]['session_time'] if 'session_time' in
                                        df_to_be_extracted['data'][i] else np.nan for i in range(len(df_to_be_extracted))]
            self.df['path'] =  [df_to_be_extracted['data'][i]['path'] if 'path' in df_to_be_extracted['data'][i]
                                else np.nan for i in range(len(df_to_be_extracted))]
            self.df['platform'] =  [df_to_be_extracted['data'][i]['platform'] if 'platform' in
                                    df_to_be_extracted['data'][i] else np.nan for i in range(len(df_to_be_extracted))]
        except:
            print("\nError encountered!\n")

    # clean data: missing values, duplicate events, tester events.
    def _clean_df_(self, remove_NULL=False, how='any', fill_NULL=False, fill_value=0):
        self.df['event_name'] = self.df['event_name'].replace(to_replace = 'View Emote Detail',
                                                              value = 'Viewed Emote detail')
        self.df = self.df[self.df.event_name != 'Viewer']
        if remove_NULL:
            self.df = self.df.dropna(0, how)
            if fill_NULL:
                self.df = self.df.fillna(fill_value, 0)

        return self.df

    def dataprep(self, remove_NULL=False, how='any', fill_NULL=False, fill_value=0):
        '''
        data preprocessing
        Args: (optional)
             remove_NULL: False, not remove missing value; True, remove all missing values
             how: - approach to remove missing values, 'any' or 'all'
             fill_NULL: '0', not fill missing data; '1', fill missing data
             fill_value: value used to fill the missing data
        '''
        if not remove_NULL:
            self._extract_data_()
        self._clean_df_(remove_NULL, how, fill_NULL, fill_value)

    # list the number of missing data for each event
    def describe_null(self):
        notnull = [len(self.df)-i for i in self.df.count(0)]
        notnull_ptg = [1.0*(len(self.df)-i)/len(self.df) for i in self.df.count(0)]
        #self.desnull_df =pd.DataFrame(notnull_ptg, columns = self.df.columns)
        missing_df = pd.DataFrame({'total missing values': notnull,
                                   'missing percentage': notnull_ptg,
                                   'data type': self.df.dtypes},
                                    index = self.df.columns.tolist())
        return missing_df

    # load pickle data
    def load(self):
        '''
        Loads existing cleaned data from a pickled dataframe
        '''
        # connect:
        con = None
        con = psycopg2.connect(database = self.dbname, user = self.username, host='localhost', password=self.pswd)

        # query:
        sql_query = """
                    SELECT * FROM event_table;
                    """
        event_data_from_sql = pd.read_sql_query(sql_query,con)

        return event_data_from_sql

    # save data to pickle/database
    def save(self):
        '''
        save data to database and to DataFrame pickle.
        Args:
            DataFrame pickle will always be overwritten (string)
        '''

        # Pickle DataFrame as extra backup
        pickle.dump(self.df, open('event_df.pickle', "w"))

        # Write to database
        ## create a database (if it doesn't exist)
        if not database_exists(self.engine.url):
            create_database(self.engine.url)

        self.df.to_sql('event_table', self.engine, if_exists='replace')
