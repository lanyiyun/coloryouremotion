'''''''''''''''''''''''''''''''''''''''
 Module for event explorative analysis
'''''''''''''''''''''''''''''''''''''''
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# find unique event name
def find_unique_event(df):
    return list(set(df['event_name']))

# segment users who submitted emote versus who did not
def segment_emote_users(df):
    df_user_id = df.groupby('user_id').groups
    users_not_submit_emote = 0
    unique_event = find_unique_event(df)
    for i in df_user_id.values():
        # event 23 is "submit emote"
        NumOfSubmitEmote = np.sum(df.loc[i]['event_name'].isin([unique_event[24]]))
        if NumOfSubmitEmote==0:
            try:
                users_not_submit_emote = users_not_submit_emote.append(df.loc[i])
            except:
                users_not_submit_emote = df.loc[i]
    users_submit_emote = df.loc[list(set(df.index.values) - set(users_not_submit_emote.index.values))]

    return users_not_submit_emote, users_submit_emote

# find active and non_active users ID
def find_active_IDs(df):
    '''
    Args:
        df: users who submitted emote
    '''
    emote_IDs = df.groupby('user_id')
    NumOfSessions = {}
    for i in emote_IDs.groups.keys():
        NumOfSessions[i] = len(emote_IDs.get_group(i).groupby('session_time').groups)
        df_sessions = pd.DataFrame([NumOfSessions.keys(), NumOfSessions.values()]).transpose()
        df_sessions = df_sessions.sort_values(by = 1, ascending = False)
    active_user_ID = df_sessions.iloc[0:11][0]
    non_active_user_ID = df_sessions.iloc[11:][0]
    return active_user_ID, non_active_user_ID

# segment active user data versus non-active user data
def segment_active_users(users_submit_emote, active_user_ID, non_active_user_ID):
    active_df = 0
    nonactive_df = 0
    for i in range(len(active_user_ID)):
        try:
            active_df = active_df.append(users_submit_emote[users_submit_emote.user_id == active_user_ID.iloc[i]])
        except:
            active_df = users_submit_emote[users_submit_emote.user_id == active_user_ID.iloc[i]]

    for i in range(len(non_active_user_ID)):
        try:
            nonactive_df = nonactive_df.append(users_submit_emote[users_submit_emote.user_id == non_active_user_ID.iloc[i]])
        except:
            nonactive_df = users_submit_emote[users_submit_emote.user_id == non_active_user_ID.iloc[i]]

    return active_df, nonactive_df

# extract event list from users in chronological order (event type can repeat in this list)
def extract_event(df):
    event_data= []
    n = 0
    event_list = df['event_name']
    if len(event_list) == 1:
        return event_list.tolist()
    else:
        for i, j in zip(event_list, event_list[1:].tolist() + event_list[:1].tolist()):
            n += 1 # count the number of iteration
            if i != j:
                if len(event_data) == 0:   # if it's the first iteration
                    event_data.append(i)
                    event_data.append(j)
                elif n != len(event_list): # if it's not the last iteration
                    event_data.append(j)
        return event_data

# function to create pairwise event links
def map_events(df, user_type = 'active'):
    '''
    Args:
        df: clean dataframe for the school (entire dataset)
        user_type: 'active' or 'nonactive'
    '''

    [user_not_submit_emote, user_submit_emote] = segment_emote_users(df)
    [active_id, nonactive_id] = find_active_IDs(user_submit_emote)
    [active_df, nonactive_df] = segment_active_users(df, active_id, nonactive_id)

    if user_type == 'active':
        active_by_session = active_df.groupby('session_time').groups
        session_graph = []
        unique_event = find_unique_event(active_df)
        for i in active_by_session:
            if len(active_by_session[i]) > 1:
                current_events = active_df.loc[active_by_session[i]]
                session_events = extract_event(current_events)
                session_eventTOpos = [unique_event.index(j) for j in session_events]
                session_graph_temp = [(session_eventTOpos[k],session_eventTOpos[k+1]) for k in range(len(session_eventTOpos)-1)]
                session_graph = session_graph + session_graph_temp
    elif user_type == 'nonactive':
        nonactive_by_session = nonactive_df.groupby('session_time').groups
        session_graph = []
        unique_event = find_unique_event(nonactive_df)
        for i in nonactive_by_session:
            if len(nonactive_by_session[i]) > 1:
                current_events = nonactive_df.loc[nonactive_by_session[i]]
                session_events = extract_event(current_events)
                session_eventTOpos = [unique_event.index(j) for j in session_events]
                session_graph_temp = [(session_eventTOpos[k],session_eventTOpos[k+1]) for k in range(len(session_eventTOpos)-1)]
                session_graph = session_graph + session_graph_temp

    return session_graph

# function to define user-app interaction network: layout, nodes, edges, etc
def define_webnet(G, graph, dict_eoccur, edge_color, interested_node = {},
               node_size = 1000, labels=None, graph_layout='shell',
               node_color='black', node_alpha=0.3,
               node_text_size=20,
               edge_alpha=0.3, edge_tickness=10,
               edge_text_pos=0.3,
               text_font='sans-serif', cmap = plt.cm.Reds):
    '''
    Args:
        G: networkx graph object
        graph:
        dict_eoccur:
        interested_node:
    '''
    # add edges
    for edge in graph:
        G.add_edge(edge[0], edge[1])

    # node layout options
    if graph_layout == 'spring':
        graph_pos=nx.spring_layout(G)
    elif graph_layout == 'spectral':
        graph_pos=nx.spectral_layout(G)
    elif graph_layout == 'random':
        graph_pos=nx.random_layout(G)
    elif graph_layout == 'circular':
        graph_pos=nx.circular_layout(G)
    else:
        graph_pos=nx.shell_layout(G)

    # scale edge width to show traffic volume
    edge_width = {}
    for i in G.edges():
        if i in dict_eoccur.keys():
            edge_width[i] = dict_eoccur[i]
    edge_width = [edge_width[i] for i in G.edges()]
    edge_width = [10.0*edge_width[i]/max(edge_width) for i in range(len(edge_width))]

    # highlight interested traffic by redefining edge_width
    if len(interested_node):
        interested_df = pd.DataFrame([edge_width, G.edges()]).transpose()

        for i in range(len(interested_df)):
            if not len(interested_node.intersection(interested_df[1][i])):
                interested_df[0][i] = 0
            if 3 in interested_df[1][i] or 7 in interested_df[1][i] or 37 in interested_df[1][i]:
                interested_df[0][i] = 0

        edge_width = interested_df[0]
        # convert to conditional values
        edge_width = [50*edge_width[i]/sum(edge_width) for i in range(len(edge_width))]

    # draw graph
    nx.draw_networkx_nodes(G,graph_pos,node_size = node_size,
                           alpha=node_alpha, node_color=node_color, cmap = cmap)
    nx.draw_networkx_edges(G,graph_pos,width=edge_width,
                           alpha=edge_alpha, edge_color=edge_color, arrows = True)
    nx.draw_networkx_labels(G, graph_pos, font_size=node_text_size,
                            font_family=text_font)

    # label nodes
    if labels is None:
        labels = range(len(graph))

    # label edges
    edge_labels = dict(zip(graph, labels))

# function to draw user-app interaction network: layout, nodes, edges, etc
def draw_webnet(df, user_type = 'active', traffic_direction = True, interested_node = {}):
    '''
    Args:
        df: clean dataframe for the school
        user_type: 'active': active users
                   'nonactive': nonactive users
        traffic_direction: True: positive direction (e.g., 9->13),
                           False: negative direction (e.g., 13->9)
    '''
    session_graph = map_events(df, user_type)

    # create networkx graph
    G=nx.Graph()

    # count eventTOevent occurance
    event_occurance = [[i,session_graph.count(i)] for i in set(session_graph)]

    # change to dict
    dict_eoccur = {}
    for i in event_occurance:
        dict_eoccur[i[0]] = i[1]

    # plot traffic
    dict_positive = {}
    dict_negative = {}
    for i in dict_eoccur.keys():
        if i[0]<i[1]:  # positive direction
            dict_positive.update({i:dict_eoccur[i]})
        elif i[0]>i[1]:  # positive direction
            dict_negative.update({i:dict_eoccur[i]})

    # split bi-directional traffic
    positive_graph = [session_graph[i] for i in range(len(session_graph)) if session_graph[i][0] < session_graph[i][1]]
    negative_graph = [session_graph[i] for i in range(len(session_graph)) if session_graph[i][0] > session_graph[i][1]]

    # swap tuple element in the negative case for the matching later
    dict_negative_revise = {}
    for i in range(len(dict_negative.keys())):
        dict_negative_revise[(dict_negative.keys()[i][1],dict_negative.keys()[i][0])] = dict_negative.values()[i]

    # draw the network
    if traffic_direction == True:
        define_webnet(G, positive_graph, dict_eoccur = dict_positive, interested_node = interested_node,
                      graph_layout = 'shell', edge_color='blue')
    elif traffic_direction == False:
        define_webnet(G, negative_graph, dict_eoccur = dict_negative_revise, interested_node = interested_node,
                      graph_layout = 'shell', edge_color='red')

    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.show()

# calculate stay_time in each event, key: event index, value: time in seconds
def cal_stay_time(df_user):
    '''
    Args:
        df_user: data frames of users who are active or not
    '''
    stay_time = {}

    unique_event = find_unique_event(df_user)
    submit_user_events = df_user.groupby('event_name').groups.keys()

    for i in unique_event:
        # if the event is in the list of user interaction
        if i in submit_user_events:
            df_event_name = df_user.groupby('event_name').groups
            df_event_in_session = df_user.loc[df_event_name[i]].groupby('session_time').groups
            # sum all the time
            for j in range(len(df_event_in_session)):
                time = df_user.loc[df_event_in_session.values()[j][-1]]['time']
                session_time = df_user.loc[df_event_in_session.values()[j][-1]]['session_time']
                if j == 0:
                    total_session_time = np.floor(0.001*(int(time) - int(session_time)))
                else:
                    total_session_time += np.floor(0.001*(int(time) - int(session_time)))
            stay_time[unique_event.index(i)] = total_session_time
        # if not, assign nan
        else:
            stay_time[unique_event.index(i)] = np.nan

    return stay_time

# function to draw time spent on each event for both active and non-active users
def draw_stay_time(active_df, nonactive_df):

    stay_time_active = cal_stay_time(active_df)
    stay_time_nonactive = cal_stay_time(nonactive_df)
    activeMeans = [stay_time_active.values()[i]/np.nansum(stay_time_active.values())
                   for i in range(len(stay_time_active))]
    nonactiveMeans = [stay_time_nonactive.values()[i]/np.nansum(stay_time_nonactive.values())
                      for i in range(len(stay_time_nonactive))]

    df_active = pd.DataFrame([activeMeans, nonactiveMeans]).transpose()
    df_active_sort = df_active.sort_values(by= 0, ascending = False)
    df_active_sort = df_active_sort.fillna(value=0)

    ind = np.arange(len(df_active_sort))  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, df_active_sort[0], width, color='b')

    rects2 = ax.bar(ind + width, df_active_sort[1], width, color='g')
    plt.xticks(range(len(df_active_sort)), df_active_sort.index)

    coefficients = np.polyfit(range(len(df_active_sort)), df_active_sort[0], deg = 6)
    polynomial = np.poly1d(coefficients)
    xs = np.arange(0, len(df_active_sort), 1)
    ys = polynomial(xs)
    line1, = plt.plot(xs, ys, linewidth = 2)

    coefficients = np.polyfit(range(len(df_active_sort)), df_active_sort[1], deg = 6)
    polynomial = np.poly1d(coefficients)
    xs = np.arange(0, len(df_active_sort), 1)
    ys = polynomial(xs)
    line2, = plt.plot(xs, ys, linewidth = 2)

    plt.legend([line1, line2], ['Active Users', 'NonActive Users'], fontsize = 14)
    plt.ylabel('stay time (%)')
    plt.xlabel('event number')
    plt.show()
