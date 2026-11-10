"""Train-only preprocessing for baseline A and engineered-feature flow B."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from .data import CATEGORICAL, NUMERIC


def derived_features(frame):
    """Age bands and admission-minus-previous-grade, calculated per row."""
    age = frame['Age at enrollment'].to_numpy(dtype=float)
    delta = (frame['Admission grade'] - frame['Previous qualification (grade)']).to_numpy(dtype=float)
    age_band = np.where(np.isnan(age), np.nan, np.select([age < 20, age < 25, age < 30], [0., 1., 2.], default=3.))
    return np.column_stack([age_band, delta])


def derived_names(transformer, input_features):
    return np.array(['age_band', 'admission_grade_difference'], dtype=object)


def build_preprocessor(flow):
    if flow not in ('A', 'B'):
        raise ValueError(f'Unknown flow {flow}')
    numeric = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    encoder = OneHotEncoder(handle_unknown='ignore') if flow == 'A' else OneHotEncoder(min_frequency=0.01, handle_unknown='infrequent_if_exist')
    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('onehot', encoder)])
    transformers = [('numeric', numeric, NUMERIC), ('categorical', categorical, CATEGORICAL)]
    if flow == 'B':
        engineered = Pipeline([('derive', FunctionTransformer(derived_features, feature_names_out=derived_names)),
                               ('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
        transformers.append(('engineered', engineered, ['Age at enrollment', 'Admission grade', 'Previous qualification (grade)']))
    return ColumnTransformer(transformers)
