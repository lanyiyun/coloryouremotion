# Color Your Emotion


####this is the code written for insight project "Color Your Emotion" available at http://coloryouremotion.com
===========================================================================================================
## An Overview of Code Structure
- The file consists four modules:
  - module 1: data preprocessing (dataprep.py)
  - module 2: explorative analysis for the event data (events.py)
  - module 3: explorative analysis for the user data (users.py)
  - module 4: prediction model (prediction.py)


===========================================================================================================
## Module 1: data preprocessing (dataprep.py)

- Description: module 1 provides defination of two classes: users and events. Instances created from these two classes                take raw data as input and will return a clean data frame that contains user and event information.



- Class Users
  - description: provide a NULL-free data frame for user information
  - methods: _read\_data_, _clean\_df_, dataprep, describe\_null, load, save


- Class Events
  - description: provide a NULL-free data frame for event information
  - methods: _read\_JSON_, _clean\_df_, _extract\_data_, dataprep, describe\_null, load, save
  

    
===========================================================================================================

## Module 2:  explorative analysis for the event data (events.py)

    
- Description: Module 2 provides function to draw traffic of user-app interaction as well as segmenting users



- Functions: 
  - find\_unique\_event, extract\_event, map\_event
  - segment\_emote\_users, find\_active\_IDs, segment\_active\_users
  - define\_webnet, draw\_webnet
  - cal\_stay\_time, draw\_stay\_time

===========================================================================================================

## Module 3:  explorative analysis for the user data (users.py)

- Description: Module 3 draws user retention of two user groups (submit an emote versus not submit an emote)


- Functions: cal\_user\_retention, draw\_user\_retention

===========================================================================================================
## Module 4:  prediction model (prediction.py)

- Description: Module 4 builds a predictive model for user drop-offs


- Functions: get\_users, sessions\_1st\_week, events\_1st\_week, define\_features, norm\_data, feature\_importance,                prediction


