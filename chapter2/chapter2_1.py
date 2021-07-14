import pandas as pd
import numpy as np 
import sklearn
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from copy import copy

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

df = pd.read_csv("datasets/housing/housing.csv")
num_attribs = copy(list(df.columns))
num_attribs.remove("ocean_proximity")
cat_attribs = ["ocean_proximity"]


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])  

df_num = full_pipeline.fit_transform(df)
new_columns = num_attribs + ['ocean_proximity_1', 'ocean_proximity_2', 'ocean_proximity_3', 'ocean_proximity_4', 'ocean_proximity_5']
df_num = pd.DataFrame(df_num, columns = new_columns)

param_grid = [
    {'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto'], 'C': [1, 10, 20]}
]
train_set, test_set = split_train_test(df_num, 0.2)
X_train = train_set.drop("median_house_value", axis=1)
y_train = train_set["median_house_value"]
print(X_train.head())
print(y_train.head())

clf = SVR()
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

regressor = SVR(kernel = 'rbf', gamma = 'scale', C = 1.0)
# degree = 3    
# df = pd.DataFrame(my_array, columns = ['Column_A','Column_B','Column_C'])

# git init
# git add --all
# git commit -m "folan"
# git remote add origin https://github.com/alirezatwk/Hands-on.git
# git branch -M main
# git push -u origin main