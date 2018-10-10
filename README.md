
#   DisasterResponse Project
-   The project's target is to analysis the message data for disaster response.

##  Installation 
- The code should run with no issues using Python versions 3.6+.
- Better use anaconda3 env, Then run **pip install -r requirements.txt** at root path,it will auto install requirement pip packages.



###   Project Infomation
-   Display directory tree:

```
./Disaster-Response-Pipeline/              --> project root directory
├── ETL\ Pipeline\ Preparation.html      --> ETL pipeline notebook export HTML 
├── ETL\ Pipeline\ Preparation.ipynb     --> ETL pipeline notebook
├── InsertDatabaseName.db                --> ETL auto created
├── LICENSE
├── ML\ Pipeline\ Preparation.html        --> ML pipeline notebook export HTML 
├── ML\ Pipeline\ Preparation.ipynb       --> ML pipeline notebook
├── README.md
├── clf.pickle
├── data                                 --> dataset directory for notebooks
│   ├── categories.csv
│   └── messages.csv
├── requirements.txt                     --> install requirement file 
└── web                                  --> flask web app directory
    ├── app
    │   ├── run.py                       --> web main run file
    │   └── templates                    --> templates files
    │       ├── go.html
    │       └── master.html
    ├── data                             --> dataset directory for pipline
    │   ├── DisasterResponse.db
    │   ├── disaster_categories.csv
    │   ├── disaster_messages.csv
    │   └── process_data.py             --> process data for ETL pipeline
    └── models                          --> models directory
        ├── classifier.pkl
        └── train_classifier.py         --> train_classifier for ML pipeline
```

### How To Run ?

- Run the following commands in the project's root directory to set up your database and model.
1. cd web directory as root path.

2. To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
3. To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

4. Run the following command in the app's directory to run your web app.
    `python run.py`

5. Go to http://0.0.0.0:3001/