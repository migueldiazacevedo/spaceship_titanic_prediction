# Spaceship Titanic 
### A new spin on an old classic machine learning challenge<br>-- the Titanic dataset
The goal of this project is to develop machine learning models to compete in the 
[Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) competition on Kaggle. 

The story of "Spaceship Titanic" is as follows:
It is the year 2912 and Spaceship Titanic was an interstellar passenger liner launched a month ago. 
En route to its first destination, the Spaceship Titanic collided with a spacetime anomaly hidden within a dust cloud. 
Sadly, it met a similar fate as its namesake from 1000 years before. 
Though the ship stayed intact, almost half of the passengers were transported to an alternate dimension!

My goal is to help rescue crews and retrieve the lost passengers by predicting which passengers were transported 
by the anomaly using records recovered from the spaceship’s damaged computer system.


## My approach:
1. Explore and process data by statistically assessing features and creating new features as necessary 
   __see data_preprocessing_EDA.ipynb for details__
2. Use preprocessed data in an automated machine learning pipeline to see viable models
   __see autoML_pycaret.ipynb for details__
3. Tune a variety of models (__see light_gbm.ipynb, random_forest.ipynb, logistic_regression.ipynb__)
4. Place all tuned models into an ensemble voting model (__see voting_model.ipynb__)
5. Submit results to Kaggle
6. Deploy app using FastAPI

## Project Files:
- __data__ <br>(a directory created in "data_preprocessing_EDA.ipynb")
  - train.csv <br>(training data)
  - test.csv <br>(testing data)
  - sample_submission.csv <br>(template for submitting data to Kaggle)
  - train_processed.pkl <br>(training data with engineered features from preprocessing pipeline)
  - test_processed.pkl <br>(test data treated with preprocessing pipeline)
  - submission.csv <br>(my final submission)
- __ML_models_trained__ <br> 
(a directory containing trained machine learning models that are sent to the final voting model)
- __notebooks__ <br>
  (a directory containing Jupyter notebooks)
  - data_preprocessing_EDA.ipynb <br>
    (***important!*** primary notebook for feature engineering and data pre-processing in Pandas)
  - ML_models <br>
    (directory containing a notebook for each machine learning model developed)
    - autoML_pycaret.ipynb <br>
      (an autoML tool called PyCaret)
    - light_gbm.ipynb <br>
      (LightGBM model) 
    - logistic_regression.ipynb<br>
      (Sklearn logistic regression model)
    - random_forest.ipynb<br>
      (Sklearn random forest model)
    - voting_model.ipynb
      (Sklearn voting ensemble model -- final model used and submitted)
- __utils__ <br>
  (a Python Module containing utility functions)
  - __ init__.py
  - data_preprocessing.py
  - machine_learning.py
- environment.yml <br>
  (.yml file for recreating conda environment)
- __deployment__<br>
  - Dockerfile
- LICENSE<br>
  (MIT License)

## How to run notebooks:
__set up environment__ <br>
First, ensure that your system has conda (I recommend downloading according to directions on www.anaconda.com)
<br>
1. Clone the repository 
   
2. Create a virtual environment using the requirements.txt file provided<br>
   e.g. 
```bash
python3 -m venv spaceship_titanic/
```
3. activate the venv and install all requirements provided 
```bash
source spaceship_titanic/bin/activate
pip install -r requirements.txt
```
4. Open the Jupyter notebook file, podcast_reviews.ipynb, in your Jupyter environment and step through to see analysis.
   

using this environment you should be able to run any notebooks for other code in this project 
### Dependencies 
  - python 3.11
  - lightgbm
  - matplotlib
  - numpy
  - optuna
  - pandas
  - seaborn
  - scikit-learn
  - scipy
  - xgboost
  - pycaret 3.3.2 <br><br>
see environment.yml for more details

### Future Directions
- more in-depth hyperparameter tuning
- Try to create more selective features by assessing pairwise relationships between existing features
- consider more types of models
- More in-depth feature engineering
- Use Sklearn pipelines and column transformers instead of pandas-based approach 

### License
[MIT](https://choosealicense.com/licenses/mit/) 
- see LICENSE file for exact details

For any questions or issues, please contact [migueldiazacevedo@gmail.com](migueldiazacevedo@gmail.com)