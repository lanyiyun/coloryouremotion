
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# append the path where modules are located
sys.path.append("/Users/lanyiyun/Documents/Insight Project/Github")

import dataprep
import events
import users
import prediction

# create events instance
dp_events_SB = dataprep.Events('Stanbridges')

# extract events and convert them into dataframe
dp_events_SB.dataprep()

# remove all the missing values
dp_events_SB.dataprep(remove_NULL=True)

# create events instance for two other schools
dp_events_BK = dataprep.Events('Brooklyn')
dp_events_NE = dataprep.Events('NewEn')
dp_events_BK.dataprep()
dp_events_NE.dataprep()
dp_events_BK.dataprep(remove_NULL=True)
dp_events_NE.dataprep(remove_NULL=True)

# concatenate all the data into one dataframe
dp_events = dp_events_SB
dp_events.df = pd.concat([dp_events_SB.df,dp_events_BK.df,dp_events_NE.df])

# save to SQL
dp_events.save()

# plot user net
events.draw_webnet(dp_events.df, user_type = 'active', traffic_direction = False, interested_node = {})
events.draw_webnet(dp_events.df, user_type = 'nonactive', traffic_direction = False, interested_node = {})
events.draw_webnet(dp_events.df, user_type = 'active', traffic_direction = True, interested_node = {})
events.draw_webnet(dp_events.df, user_type = 'nonactive', traffic_direction = True, interested_node = {})

# draw user stay time on each event
[users_not_submit_emote, users_submit_emote] = events.segment_emote_users(dp_events.df)
[active_user_ID, non_active_user_ID] = events.find_active_IDs(users_submit_emote)
[active_df, nonactive_df] = events.segment_active_users(users_submit_emote, active_user_ID, non_active_user_ID)
events.draw_stay_time(active_df, nonactive_df)

# plot user retention
users.draw_user_retention(dp_events.df)

# obtain users (with labels) for prediction
User_4_pred = prediction.get_users(users_submit_emote)

# define and add feature column to the data frame
User_4_pred = prediction.define_features(User_4_pred, users_submit_emote)

# model for prediction
from collections import Counter
label_counts = Counter(User_4_pred.label)
dis = pd.DataFrame.from_dict(label_counts, orient='index')
dis.plot(kind='bar')

# split data for training and testing
[train_transform, train, test_transform, test] = prediction.norm_data(User_4_pred)
df_train_transform = pd.DataFrame(train_transform)
df_train_transform['label'] = train['label'].values

# plot features in each group
color = dict(boxes='Green', whiskers='Orange',
             medians='Blue', caps='Red')
fig, axes = plt.subplots(figsize=(15, 5),nrows=2, ncols=2)
df_train_transform[df_train_transform['label']==0].ix[:,0:-1].plot.box(color=color,
                   ax=axes[0,0],title='Group 1',fontsize=15)
df_train_transform[df_train_transform['label']==1].ix[:,0:-1].plot.box(color=color,
                   ax=axes[0,1],title='Group 2',fontsize=15)
df_train_transform[df_train_transform['label']==2].ix[:,0:-1].plot.box(color=color,
                   ax=axes[1,0],title='Group 3',fontsize=15)
df_train_transform[df_train_transform['label']==3].ix[:,0:-1].plot.box(color=color,
                   ax=axes[1,1],title='Group 4',fontsize=15)
plt.show()

# check feature correlation
corr = df_train_transform.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(100, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap,
            square=True,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

# plot feature Importance
prediction.feat_importance(train)

# predict with LDA
[y_pred, y_test] = prediction.prediction(train_transform, train, test_transform, test)
print classification_report(y_test, y_pred)
