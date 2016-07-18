'''''''''''''''''''''''''''
 Module for prediction
'''''''''''''''''''''''''''
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pylab import *

import sklearn.svm
import sklearn.lda
import sklearn.cross_validation
from sklearn import linear_model

from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

# get the entire data set for training and testing for one school, with labels
def get_users(df, sign = 1):

    '''
    Args:
        df is the segmented users: submit an emote VS not submit an emote.
        sign = 0 is user not submit; sign = 1 default, is user who submit an emote
    '''

    # df is the segmented users: submit an emote VS not submit an emote.
    # sign = 0 is user not submit; sign = 1 default, is user who submit an emote
    user_ID = {}
    number_of_event = {}
    time_first_seen = {}
    time_first_emote = {}
    time_last_seen = {}

    NumOfEmoteUser = len(df.groupby('user_id'))

    for k in range(NumOfEmoteUser):
        sort_user = df.loc[df.groupby(['user_id']).groups.values()[k]].sort_values(by = 'time')

        # user ID
        user_ID[k] = sort_user['user_id'].iloc[0]

        # unique events
        number_of_event[k] = len(set(sort_user['event_name']))

        # first time register
        time_first_seen[k] = sort_user['time'].iloc[0]

        # first time submit an emote
        lp = range(len(sort_user))
        for i in lp:
            extract_event = sort_user.iloc[i]['event_name']
            if extract_event == 'submit_emote':
                time_first_emote[k] = sort_user.iloc[i]['time']
                break

        #time_last_seen
        time_last_seen[k] = sort_user['time'].iloc[-1]

    if sign:
        df_time = pd.DataFrame([user_ID.values(), time_first_seen.values(), time_first_emote.values(),
                                time_last_seen.values(), number_of_event.values()])
        df_time = df_time.transpose().rename(columns = {0: 'user_ID', 1: 'first seen', 2:'first emote',
                                                        3:'last seen', 4: 'unique events'})
    else:
        df_time = pd.DataFrame([time_first_seen.values(), time_last_seen.values()])
        df_time = df_time.transpose().rename(columns = {0:'first seen', 1:'last seen'})

    # label users based on their active index
    df_time['active days'] = [0.001*(int(df_time['last seen'][i]) - int(df_time['first seen'][i]))/3600/24
                         for i in range(len(df_time))]
    df_time = df_time.sort_values(by='active days')
    df_time['active index'] = [df_time['active days'].iloc[i]/max(df_time['active days']) for i in range(len(df_time))]

    label = [np.zeros(len([x for x in df_time['active index'] if x < 0.1])).tolist() +
             np.ones(len([x for x in df_time['active index'] if x < 0.55 and x >0.1])).tolist() +
            (2*np.ones(len([x for x in df_time['active index'] if x < 0.9 and x >0.55]))).tolist() +
            (3*np.ones(len([x for x in df_time['active index'] if x > 0.9]))).tolist()]
    df_time['label'] = label[0]
    df_time = df_time.reset_index()
    df_time = df_time.drop('index', 1)

    return df_time

# function to find how many sessions in the first week
def sessions_1st_week(userID, users_submit_emote):
    submit_groups = users_submit_emote.groupby('user_id')
    all_session = sorted(submit_groups.get_group(userID).groupby('session_time').groups.keys())
    start_time = int(all_session[0])
    for i in range(len(all_session)):
        duration = 0.001*(int(all_session[i]) - start_time)/3600/24
        if duration > 7:
            break
    return i

# function to find the list of events for every user in the first week
def events_1st_week(userID, users_submit_emote):
    submit_groups = users_submit_emote.groupby('user_id')
    start_time = int(submit_groups.get_group(userID)['time'].iloc[0])
    for i in range(len(submit_groups.get_group(userID))):
        duration = 0.001*(int(submit_groups.get_group(userID)['time'].iloc[i]) - start_time)/3600/24
        if duration > 7:
            break
    return submit_groups.get_group(userID).iloc[0:i]

# function to define and add features as columns to the data frame
def define_features(df_time, users_submit_emote):

    '''
    Args:
        df_time: data frame returned from get_users
        users_submit_emote: users who submit emote
    '''

    session_1week = {}
    submit_groups = users_submit_emote.groupby('user_id')
    for i in submit_groups.groups.keys():
        session_1week[i] = sessions_1st_week(i, users_submit_emote)

    df_time['f1:session'] = [session_1week[i] for i in df_time['user_ID']]
    df_time['f1:session'] = [1.0*(df_time['f1:session'][i]-np.mean(df_time['f1:session']))/np.std(df_time['f1:session'])
                            for i in range(len(df_time))]

    same_day = [0.001*(int(df_time['first emote'][i]) - int(df_time['first seen'][i]))/3600/24
                for i in range(len(df_time['first seen']))]

    # features to add: all features are quantified based on the first week of activity
    emote_1st_week = [0 if same_day[i]>7 else 1 for i in range(len(same_day))]
    df_time['f2:1stEmote'] = emote_1st_week
    temp = users_submit_emote.groupby('user_id').groups
    NumOfEmotes = [1.0*users_submit_emote.loc[events_1st_week(i,users_submit_emote).index.values]
                   ['event_name'].tolist().count('submit_emote')/np.ceil(df_time[df_time['user_ID']==i]
                   ['active days']) for i in df_time['user_ID']]
    NumOfViewNewEmote = [1.0*users_submit_emote.loc[events_1st_week(i,users_submit_emote).index.values]
                         ['event_name'].tolist().count('Viewed new emote page 1')/np.ceil(df_time[df_time['user_ID']==i]
                         ['active days']) for i in df_time['user_ID']]
    NumOfViewEmoteDetail = [1.0*users_submit_emote.loc[events_1st_week(i,users_submit_emote).index.values]
                            ['event_name'].tolist().count('Viewed Emote detail')/np.ceil(df_time[df_time['user_ID']==i]
                            ['active days']) for i in df_time['user_ID']]
    NumOfViewEmoteOverview = [1.0*users_submit_emote.loc[events_1st_week(i,users_submit_emote).index.values]
                              ['event_name'].tolist().count('Viewed Emote overview')/np.ceil(df_time[df_time['user_ID']==i]
                              ['active days']) for i in df_time['user_ID']]
    NumOfViewCharts = [1.0*users_submit_emote.loc[events_1st_week(i,users_submit_emote).index.values]
                       ['event_name'].tolist().count('Viewed Charts')/np.ceil(df_time[df_time['user_ID']==i]
                       ['active days']) for i in df_time['user_ID']]
    NumOfTouch = [1.0*users_submit_emote.loc[events_1st_week(i,users_submit_emote).index.values]
                  ['event_name'].tolist().count('Touched free text during submission')/np.ceil(df_time[df_time['user_ID']==i]
                  ['active days']) for i in df_time['user_ID']]
    NumOfTimeline = [1.0*users_submit_emote.loc[events_1st_week(i,users_submit_emote).index.values]
                     ['event_name'].tolist().count('Viewed Timeline')/np.ceil(df_time[df_time['user_ID']==i]
                     ['active days']) for i in df_time['user_ID']]
    NumOfMiddle = [1.0*users_submit_emote.loc[events_1st_week(i,users_submit_emote).index.values]
                   ['event_name'].tolist().count('Roster: click on main middle part')/np.ceil(df_time[df_time['user_ID']==i]
                   ['active days']) for i in df_time['user_ID']]
    NumOfProfile = [1.0*users_submit_emote.loc[events_1st_week(i,users_submit_emote).index.values]
                    ['event_name'].tolist().count('Viewed student profile')/np.ceil(df_time[df_time['user_ID']==i]
                    ['active days']) for i in df_time['user_ID']]

    df_time['f3:NumOfEmotes'] = NumOfEmotes
    df_time['f4:New Emote'] = NumOfViewNewEmote
    df_time['f5:Emote Detail'] = NumOfViewEmoteDetail
    df_time['f6:Emote Overview'] = NumOfViewEmoteOverview
    df_time['f7:Charts'] = NumOfViewCharts
    df_time ['f8:Touch'] = NumOfTouch
    df_time['f9:Timeline'] = NumOfTimeline
    df_time['f10:Middle'] = NumOfMiddle
    df_time ['f11:Profile'] = NumOfProfile

    return df_time

def feat_importance(df_time):
    model = ExtraTreesClassifier()
    model.fit(df_time[['f1:session','f2:1stEmote','f3:NumOfEmotes', 'f4:New Emote', 'f5:Emote Detail',
    'f6:Emote Overview', 'f7:Charts', 'f8:Touch', 'f9:Timeline', 'f10:Middle', 'f11:Profile']], df_time['label'])
    # display the relative importance of each attribute
    print(model.feature_importances_)
    plt.barh(range(len(model.feature_importances_)), sorted(model.feature_importances_))
    plt.yticks(np.arange(11)+0.5, reversed(['Profile', 'Session', 'NumOfEmotes', 'Middle Part', 'Touch',
                                            'Emote Detail', 'Emote Overview', '1st Emote',
                                             'New Emote', 'Timeline','Touch Text']))
    title('Feature Importance', size =20)
    plt.show()

# split and standardize data
def norm_data(df_time):

    data_set = df_time[['f1:session','f2:1stEmote','f3:NumOfEmotes', 'f4:New Emote', 'f5:Emote Detail',
    'f6:Emote Overview', 'f7:Charts', 'f8:Touch', 'f9:Timeline', 'f10:Middle', 'f11:Profile','label']]
    train, test = sklearn.cross_validation.train_test_split(data_set, test_size = 0.25,
                                                            train_size = 0.75, random_state = 123)

    train_scale = sklearn.preprocessing.StandardScaler().fit(train.drop(['label'],1))
    train_transform = train_scale.transform(train.drop(['label'],1))
    test_scale = sklearn.preprocessing.StandardScaler().fit(test.drop(['label'],1))
    test_transform = train_scale.transform(test.drop(['label'],1))

    data_scale = sklearn.preprocessing.StandardScaler().fit(data_set.drop(['label'],1))
    data_transform = data_scale.transform(data_set.drop(['label'],1))

    return train_transform, train, test_transform, test

def prediction(train_transform, train, test_transform, test):
    # fit model
    ## lda model
    lda_clf = sklearn.lda.LDA()
    lda_clf.fit(train_transform, train['label'])

    # prediction
    print('1st sample from test dataset classified as:', lda_clf.predict(test_transform))
    print('actual class label:', test['label'])

    # confusion matrix
    print('Confusion Matrix of the LDA-classifier')
    print(metrics.confusion_matrix(test['label'], lda_clf.predict(test_transform)))

    y_pred = lda_clf.predict(test_transform)
    y_test = test['label']

    # Compute confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    plt.show()

    return y_pred, y_test


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(4)
    plt.xticks(tick_marks, ['1 month', '3 months', '6 months', 'over 6 months'], rotation=45)
    plt.yticks(tick_marks, ['1 month', '3 months', '6 months', 'over 6 months'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
