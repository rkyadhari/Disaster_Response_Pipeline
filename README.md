# Disaster_Response_Pipeline
**Installation****
Libraries used are :
-sys
-pickle
-re
-pandas
-numpy
-nltk
-sqlalchemy

**Project Motivation**
Project Overview:
In this project, we will apply the skills to analyze disaster data provided by Udacity in collaboration with Figure Eight to build a model for an API that classifies disaster messages.

In the Project Workspace, you'll find a data set containing real messages that were sent during disaster events. You will be creating a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

Your project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

**Project Descriptions**
The project has three componants which are:

**ETL Pipeline:** process_data.py file contain the script to create ETL pipline which:
 Loads the messages and categories datasets.
 Merges the two datasets
 Cleans the data
 Stores it in a SQLite database
 
**ML Pipeline**: train_classifier.py file contain the script to create ML pipline which:
 Loads data from the SQLite database
 Splits the dataset into training and test sets
 Builds a text processing and machine learning pipeline
 Trains and tunes a model using GridSearchCV
 Outputs results on the test set
 Exports the final model as a pickle file
 
**Flask Web App:** the web app enables the user to enter a disaster message, and then view the categories of the message.
 The web app also contains some visualizations that describe the data
 
**Files Descriptions**
README.md: read me file
\app
  run.py: flask file to run the app
  \templates
   master.html: main page of the web application
   go.html: result web page

\data
  disaster_categories.csv: categories dataset
  disaster_messages.csv: messages dataset
  DisasterResponse.db: disaster response database
  process_data.py: ETL process
  
\models
  train_classifier.py: classification code
  classifier.pkl: model pickle file

\ETL & ML Pipeline 
  ETL Pipeline Preparation.py
  ML Pipeline Preparation.py
  
**Instructions:**
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`
    
3.On another terminal, run env|grep WORK command to derive space id and sapce domain    

3. Example  "https://viewa**7a4999b**-3001.udacity-student-workspaces.com/ - changing the highlighted values with individual space id will launch the app on the browser

**Acknowledgements**
Thanks to Udacity and Figure Eight  for providing the dataset and giving us an opportunity to explore this data set

**Output Visuals**
![Disaster Response Web App](https://user-images.githubusercontent.com/85522420/123527415-3190a900-d694-11eb-84b3-8a9ab41110ac.png)
![Graph1](https://user-images.githubusercontent.com/85522420/123527431-57b64900-d694-11eb-922f-ce0975f0dd29.png)
![Distribution of message category-graph](https://user-images.githubusercontent.com/85522420/123527433-60a71a80-d694-11eb-98fe-7ff4cc53cc03.png)

Executing process.py
![Screenshot-Executing Process py](https://user-images.githubusercontent.com/85522420/123527453-98ae5d80-d694-11eb-8498-5e088f369b21.png)

Executing train_classifier.py
![Screenshot -train_classifier - 1](https://user-images.githubusercontent.com/85522420/123527462-a95ed380-d694-11eb-9030-04e034eba7c0.png)
![Screenshot -train_classifier - 3](https://user-images.githubusercontent.com/85522420/123527465-af54b480-d694-11eb-8bd3-8cf58f2c47b8.png)



