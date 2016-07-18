'''''''''''''''''''''''''''''''''''''''
 Module for user explorative analysis
'''''''''''''''''''''''''''''''''''''''
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pylab import *
import events

# aligned array addition (same length)
def add_array(a, b):
    if len(a) < len(b):
        c = b.copy()
        c[:len(a)] += a
    else:
        c = a.copy()
        c[:len(b)] += b
    return c

# create array for user retention
def user_array(df):
    for i in range(len(df)):
        if i == 0:
            sum = np.ones(np.ceil(df)[i])
        else:
            sum = add_array(sum,np.ones(np.ceil(df)[i]))
    return sum

#user_array(df_user_not_submit)
def user_normalize(raw):
    return [float(i)/raw[0] for i in raw]

# sort events per user in choronological order
def user_retention(df, sign = 1):
    # df is the segmented users: submit an emote VS not submit an emote.
    # sign = 0 is user not submit; sign = 1 default, is user who submit an emote
    time_first_seen = {}
    time_first_emote = {}
    time_last_seen = {}
    NumOfEmoteUser = len(df.groupby('user_id'))

    for k in range(NumOfEmoteUser):
        sort_user = df.loc[df.groupby(['user_id']).groups.values()[k]].sort_values(by = 'time')

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
        df_time = pd.DataFrame([time_first_seen.values(), time_first_emote.values(), time_last_seen.values()])
        df_time = df_time.transpose().rename(columns = {0:'first seen', 1:'first emote', 2:'last seen'})
    else:
        df_time = pd.DataFrame([time_first_seen.values(), time_last_seen.values()])
        df_time = df_time.transpose().rename(columns = {0:'first seen', 1:'last seen'})

    return df_time

# draw user retention treand between users who submitted verus did not submit emote
def draw_user_retention(df):
    [user_not_submit_emote, user_submit_emote] = events.segment_emote_users(df)

    # for users who did not submit an emote
    df_user_not_submit = user_retention(user_not_submit_emote, 0)
    days_not_user_stay = [0.001*(int(df_user_not_submit['last seen'][i]) -
                                 int(df_user_not_submit['first seen'][i]))/3600/24
                            for i in range(len(df_user_not_submit['first seen']))]
    days_not_user_stay = [days_not_user_stay[i]+1 for i in range(len(days_not_user_stay))]

    # for users who submit an emote
    df_user_submit = user_retention(user_submit_emote)
    days_user_stay = [0.001*(int(df_user_submit['last seen'][i]) - int(df_user_submit['first seen'][i]))/3600/24
                        for i in range(len(df_user_submit['first seen']))]

    line1, = plt.plot(range(len(user_array(days_not_user_stay))),
                      user_normalize(user_array(days_user_stay)[0:len(user_array(days_not_user_stay))]),
                      color = 'blue', linewidth=6)
    line2, = plt.plot(range(len(user_array(days_not_user_stay))), user_normalize(user_array(days_not_user_stay)),
                      linewidth=6, color = 'red')

    plt.legend([line1, line2], ['Users who have submit emote', 'Users who have not submit emote'],
                fontsize = 14, loc = 4)
    plt.xlabel('Number of Days Remain Active', fontsize=18)
    plt.ylabel('Percentage of Users', fontsize=18)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.title('User Retention 2015-2016', fontsize=18)
    plt.show()
