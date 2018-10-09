import sys
import re
import numpy as np
import pandas as pd
import nltk
import pickle

from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report,f1_score,accuracy_score,log_loss

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

def load_data(database_filepath):
    """
    load data from database
    :param database_filepath: database file path
    :return: (X,Y,Y_labels)
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('InsertTableName', engine)

    Y_labels = ['related', 'request', 'offer', 'aid_related',
                'medical_help', 'medical_products', 'search_and_rescue',
                'security', 'military', 'child_alone', 'water', 'food',
                'shelter', 'clothing', 'money', 'missing_people', 'refugees',
                'death', 'other_aid', 'infrastructure_related', 'transport',
                'buildings', 'electricity', 'tools', 'hospitals', 'shops',
                'aid_centers', 'other_infrastructure', 'weather_related',
                'floods', 'storm', 'fire', 'earthquake', 'cold',
                'other_weather', 'direct_report']
    X = df['message'].values
    Y = df[Y_labels].values

    return (X,Y,Y_labels)


def tokenize(text):
    """
    process text, get tokenize
    :param text: text
    :return: tokens
    """
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    # remove punctation
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return (tokens)


def build_model():
    """
    build ML model
    :return: cv
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        #     ('clf', MultiOutputClassifier(RandomForestClassifier(random_state = 42), n_jobs = -1)),
        #     ('clf', MultiOutputClassifier(KNeighborsClassifier(), n_jobs = -1)),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(random_state=42), n_jobs=-1))
    ])

    parameters = {
        'vect__min_df': [1],
        'vect__lowercase': [False],
        'tfidf__smooth_idf': [False],
    }

    cv = GridSearchCV(pipeline, parameters, cv=2, n_jobs=-1)

    return (cv)


def evaluate_model(model, X_test, Y_test, category_names):
    """
    evaluate model
    :param model: ML model
    :param X_test: data
    :param Y_test: label
    :param category_names: category names
    :return: None
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test[:, 1], y_pred[:, 1], target_names=category_names))
    print('accuracy_score:{}'.format(accuracy_score(Y_test[:, 1], y_pred[:, 1])))
    print('log_loss:{}'.format(log_loss(Y_test[:, 1], y_pred[:, 1])))


def save_model(model, model_filepath):
    """
    save result model
    :param model: model
    :param model_filepath: save path
    :return:
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


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