import numpy as np
import pandas as pd
from feature_engine.encoding import (OrdinalEncoder, OneHotEncoder)
from feature_engine.transformation import (YeoJohnsonTransformer)
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score

RANDOM_STATE = 0
TEST_SIZE = 0.3

VARS_TO_DROP = ['customerID']
CAT_VARS_ONEHOT = [
    'gender', 'Partner', 'Dependents',
    'PhoneService', 'PaperlessBilling']
CAT_VARS_ORDINAL_ARBITARY = [
    'MultipleLines', 'InternetService', 'OnlineSecurity', 
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
NUM_VARS_YEO_YOHNSON = ['TotalCharges']
TARGET = 'LUNG_CANCER'
VAR_REPLACE_EMPTY_DATA = ['TotalCharges']
VARS_HOM = [ 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY','PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING','ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH','SWALLOWING DIFFICULTY', 'CHEST PAIN']
VAR_GENDER = 'GENDER'

VARS = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY','PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING','ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH','SWALLOWING DIFFICULTY', 'CHEST PAIN']


def check_keys(dict_, required_keys):
    """
    Check if a dictionary contains all expected keys
    """
    for key in required_keys:
        if key not in dict_:
            raise ValueError(f'input argument "data_files" is missing required key "{key}"')


def get_read_data(filepath) -> pd.DataFrame:
    return pd.read_csv(filepath)


def homologate_values_dummy(data):
    for x in VARS_HOM:
        data[x]=data[x].replace([1,2],[0,1])
    data[VAR_GENDER] = data[VAR_GENDER].replace(["M","F"],[1,0])
    data[TARGET] = data[TARGET].replace(["YES","NO"],[1,0])

    return data

def get_split_train_test(data):

    x_train, x_test, y_train, y_test = train_test_split(
        data.drop([TARGET], axis=1),
        data[TARGET],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    return x_train, x_test, y_train, y_test



def replace_empty_in_col(data: pd.DataFrame) -> pd.DataFrame:
    """
    In order to convert a string variable that is numeric to float,
    replace empty space value with -1
    """
    for feature in VAR_REPLACE_EMPTY_DATA:
        data[feature] = data[feature].str.replace(' ', '-1').astype(float)

    return data


def fit_categorical_encoders(x_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Fit categorical encoders on train data
    """
    ordinal_encoder_arbitrary = OrdinalEncoder(encoding_method='arbitrary', variables=CAT_VARS_ORDINAL_ARBITARY)
    ordinal_encoder_arbitrary.fit(x_train, y_train)

    onehot_encoder = OneHotEncoder(variables=CAT_VARS_ONEHOT)
    onehot_encoder.fit(x_train)

    cat_encoders = {'ordinal_encoder': ordinal_encoder_arbitrary,
                    'onehot_encoder': onehot_encoder}

    return cat_encoders


def transform_categorical_encoders(x_to_encode: pd.DataFrame, cat_encoders: dict) -> pd.DataFrame:
    """
    Use pre-fitted categorical encoders to transform data
    """
    for encoder in cat_encoders.values():
        x_to_encode = encoder.transform(x_to_encode)

    return x_to_encode


def fit_numerical_transformers(x_train: pd.DataFrame) -> dict:
    """
    Fit numerical transformers on train data
    """
    yeo_transformer = YeoJohnsonTransformer(variables=NUM_VARS_YEO_YOHNSON)
    yeo_transformer.fit(x_train)

    num_transformers = {'yeo_transformer': yeo_transformer}

    return num_transformers


def transform_numerical_transformers(x_to_transform: pd.DataFrame, num_transformers: dict) -> pd.DataFrame:
    """
    Use pre-fitted numerical transformers to transform data
    :return:
    """
    for transformer in num_transformers.values():
        x_to_transform = transformer.transform(x_to_transform)

    return x_to_transform


def fit_target_encoder(y_train: pd.Series):
    """
    Fit an encoder for the target variable
    """
    le = LabelEncoder()
    le.fit(y_train)

    return le


def transform_target_encoder(encoder, y_to_transform: pd.Series) -> pd.Series:
    """
    Use a pre-fitted encoder to transform a Series
    """
    y_to_transform = encoder.transform(y_to_transform)

    return y_to_transform


def fit_data_scaler(x_train: pd.DataFrame):
    """
    Fit a scaler to normalize data
    """
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(x_train)

    return min_max_scaler


def transform_data_scaler(scaler, x_to_transform: pd.DataFrame) -> pd.DataFrame:
    """
    Use a ore-fitted scaler to normalise data
    """
    x_to_transform = pd.DataFrame(scaler.transform(x_to_transform), columns=x_to_transform.columns)

    return x_to_transform


def oversample_data(x_train: pd.DataFrame, y_train: pd.Series):
    """
    Create artificial rows so that both classes have equal observations
    """
    x_train, y_train = SMOTE(random_state=RANDOM_STATE).fit_resample(x_train, y_train)

    return x_train, y_train


def get_preprocess_data(x_train, x_test, y_train, y_test):

    #x_train = replace_empty_in_col(x_train)
    #x_test = replace_empty_in_col(x_test)

    #cat_encoders = fit_categorical_encoders(x_train, y_train)
    #x_train = transform_categorical_encoders(x_train, cat_encoders)
    #x_test = transform_categorical_encoders(x_test, cat_encoders)

    #num_transformers = fit_numerical_transformers(x_train)
    #x_train = transform_numerical_transformers(x_train, num_transformers)
    #x_test = transform_numerical_transformers(x_test, num_transformers)

    #target_encoder = fit_target_encoder(y_train)
    #y_train = transform_target_encoder(target_encoder, y_train)
    #y_test = transform_target_encoder(target_encoder, y_test)

    #scaler = fit_data_scaler(x_train)
    #x_train = transform_data_scaler(scaler, x_train)
    #x_test = transform_data_scaler(scaler, x_test)

    x_train, y_train = oversample_data(x_train, y_train)

    return x_train, x_test, y_train, y_test


def get_model_1():
    """
    Define a model and return it
    """
    #accepted_models = ['logistic_regression', 'XGB']
    #if model_name not in accepted_models:
    #    raise ValueError(f"'model name' should be in: {accepted_models}.")

    #model = None
    #if model_name == 'logistic_regression':
    C = 0.7
    iterations = 200
    model = LogisticRegression(C=C, max_iter=iterations)

    return model
def objective(trial, xtrain, x_valid, ytrain, y_valid):
    # Definir los rangos de búsqueda de los hiperparámetros
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    max_depth = trial.suggest_int('max_depth', 2, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    
    # Crear un clasificador Random Forest con los hiperparámetros seleccionados
    clf = RandomForestClassifier(n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,
                                  random_state=42)
    
    # Entrenar y evaluar el modelo
    clf.fit(xtrain, ytrain)
    y_pred = clf.predict(x_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    
    return accuracy

def get_model_2(xtrain,x_valid, ytrain,y_valid):
    """
    Define a model and return it
    """
    
    # Crear un objeto de estudio Optuna
    study = optuna.create_study(direction='maximize')

    # Ejecutar la optimización de hiperparámetros
    study.optimize(lambda trial: objective(trial, xtrain, x_valid, ytrain, y_valid), n_trials=100)

    # Obtener los hiperparámetros óptimos
    best_params = study.best_params
    print(f'Hiperparámetros óptimos: {best_params}')

    model = RandomForestClassifier(**best_params)

    return model

def get_val_performance(y_true: np.array, y_pred: np.array):
    """
    Get performance metrics for the validation set
    """
    val_accuracy = accuracy_score(y_true, y_pred)
    val_f1 = f1_score(y_true, y_pred, average='macro')

    return val_accuracy, val_f1


def get_cv_performance(x_train: pd.DataFrame, y_train: np.array, model):
    """
    Get performance metrics for the train set using cross validation
    """
    # For CV model should be sklearn and not a loaded trained mlflow model
    if "sklearn" not in str(type(model)):
        raise TypeError("Model should be and sklearn model.")
    cv_accuracy = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy').mean()
    cv_f1 = cross_val_score(model, x_train, y_train, cv=5, scoring='f1_macro').mean()

    return cv_accuracy, cv_f1


def get_predictions(x_test: pd.DataFrame, model):
    """
    Use a model to predict on given data
    """
    y_pred = model.predict(x_test)

    return y_pred