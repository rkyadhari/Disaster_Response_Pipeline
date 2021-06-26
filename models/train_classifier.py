"""
Classifier Trainer
Script Syntax:
> python train_classifier.py <path to sqllite  destination db> <path to the pickle file>
Script Execution:
> python train_classifier.py ../disaster_response_pipeline_project/data/DisasterResponse classifier.pkl
Arguments:
    1) Path to SQLite destination database (e.g. DisasterResponse.db)
    2) Path to pickle file name where ML model needs to be saved (e.g. classifier.pkl)
"""
# import libraries
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# import libraries
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)

import sys, pickle, re
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator,TransformerMixin
from sqlalchemy import create_engine

def load_data(database_filepath):
    """
    Load Data from the Database Function
    
    Fucntion to load the database from the given filepath and process them as X, y and category_names
    Input: Path to SQLite destination database (database filepath)
    Output:
        X -> a dataframe containing features
        Y -> a dataframe containing labels
        category_names -> List of categories name
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("msgcategory", engine)
    # related field has 3 values 0,1,2 and 2 contains minimal values. To avoid value error during modleing, as suggested
    # in the knowledge forum by Udacity mentor, we will replace 2 such that the related field just contains 2 values 0,1
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns 
    return X, y, category_names

def tokenize(text,url_place_holder_string="urlplaceholder"):
    """
    Function to tokenize the text messages
    Input: text
    output: cleaned tokenized text as a list object
    """
    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Extract all the urls from the provided text 
    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        text = text.replace(url,url_place_holder_string)
        
    # Extract the word tokens from the provided text
    tokens = word_tokenize(text)
    
    #Lemmanitizer to remove inflectional and derivationally related forms of a word
    lemmatizer = WordNetLemmatizer()

    # List of clean tokens
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

#Build a custom transformer which will extract the starting verb of a sentence
class StartingVerbExtractor (BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """
    
    def starting_verb (self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB','VBP'] or first_word == 'RT':
                return True
        return False
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    """
    Function to build a model, create pipeline, hypertuning as well as gridsearchcv
    Input: N/A
    Output: Returns the model
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
        ('text_pipeline', Pipeline([   
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer())
        ])),
        ])),
        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
        ])
    
    parameters = {
       'classifier__estimator__learning_rate': [0.01, 0.02, 0.05],
       'classifier__estimator__n_estimators': [10, 20, 40]

                                 
    }    
    cv = GridSearchCV(pipeline, param_grid=parameters,scoring='f1_micro', n_jobs=-1)

     
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function to evaluate a model and return the accurancy score.
    Inputs: Model, X_test, y_test, Catgegory_names
    Outputs: Prints the Classification report & Accuracy Score
    """
    y_pred = model.predict(X_test)
    # Print the whole classification report.
    print(classification_report(y_pred, Y_test.values, target_names=category_names))
    # print raw accuracy score 
    print('Accuracy Score: {}'.format(np.mean(Y_test.values == y_pred)))


def save_model(model, model_filepath):
    """
     Save model function
    
     This function saves trained model as Pickle file, to be loaded later.
    
      Arguments:
        model -> GridSearchCV  object
        model_filepath -> destination path to save .pkl file
    
    """
    with open(model_filepath, 'wb') as file:
         pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()