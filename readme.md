# Project Description
This is a MLOps project to predict Diabetes (0 indicating negative & 1 indicating posotive) from a set of parameters like Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function & Age.<br />
The final app is being deployed to Heroku. Data pipelining and versioning is done using open-source tool, DVC. The CICD pipeline is set up using Github Workflows Action.<br/>

### Dataset Context
Dataset can be downloaded from https://www.kaggle.com/uciml/pima-indians-diabetes-database

This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

# Steps Followed in creating the mlops project

### create a virtual environment inside the current working directory

```bash
conda create --prefix=environment_name python=3.7 -y
```

### activate the environment form the folder where its created
```bash
conda activate environment_name
```

### create a requirements.txt file, mention all dependencies & install them
```bash
pip install -r requirements.txt
```

### create a project structure
```bash
python template.py
```

### Keep the raw dataset in /source_data/

### Initialize git repo
```bash
git init
```

### Initialize DVC repo
```bash
dvc init
```

### Add dataset to DVC for tracking
```bash
dvc add source_data/diabetes.csv
```

### Add project & model configurations in params.yaml

### Add pipeline for fetching data from Data source- /src/get_data.py

### Add pipeline for loading the raw data locally- /src/load_data.py

### Add load_data stage in dvc.yaml & then run
```bash
dvc repro
```
to run the pipeline

### Add pipeline for spliting dataset into train and test data- /src/split_data.py

### Add split_data stage in dvc.yaml file & then run
```bash
dvc repro
```
to run the pipeline

### Add pipeline for training and evaluation- /src/train_and_evaluate.py

### Add train_and_evaluate stage in dvc.yaml file & then run
```bash
dvc repro
```
to run the pipeline
<br />
see that the trained model has got saved in /saved_models <br />
Copy the saved model to /prediction_service/model


### To see current parameters ans scores, run
```bash
dvc metrics show
```

### Change model paramerters and run
```bash
dvc repro
```
to run the pipeline again

### To compare previous and current parameters and scores, run
```bash
dvc metrics diff
```

### Get the input range for all features
Refer to notebooks/Test.ipynb
Copy the generated schema_in.json to /tests and /prediction_service


### Design test cases for input validation
Refer to -
/prediction_service/prediction.py
/tests/test_config.py

# To run all test cases in the current environment, run-
```bash
pytest -v
```

### Create and configure tox.ini file and add different environments to carry out tests on
To run tests on customized environments, ensure test cases are added to /tests/test_config.py, & run-
```bash
tox
```

### Create setup.py file & add packaging info to it
Then run-
```bash
pip install -e .
```
We would see a folder in root dir named {name}.egg-info, where we find Packaging info and contents to be packaged.

Note: To create a distribution & package as a tar, run-
```bash
python setup.py sdist bdist_wheel
```
It would create a distribution for us in /dist, which we can share with others for installing the library.

### Design the frontend of the webapp
Refer to /webapp/

### Write the backend webapp code in app.py
To run the flask application, run-
```bash
python app.py
```

### Set up a CICD pipeline using github workflows action
Refer to /.github/workflows/ci-cd.yaml
<br />
Push changes to github and navigate to github actions to see the build status
