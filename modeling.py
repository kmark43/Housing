import pandas as pd
from sklearn.preprocessing import *
from sklearn.pipeline import *
from sklearn.impute import *
from sklearn.compose import *
from sklearn.model_selection import *
from sklearn.metrics import *
from xgboost import XGBRegressor

# Suburb	Address	Rooms	Type	Price	Method	SellerG	Date	
# Distance	Postcode	Bedroom2	Bathroom	Car	Landsize	
# BuildingArea	YearBuilt	CouncilArea	Lattitude	
# Longtitude	Regionname	Propertycount

# Suburb, Address->street, Rooms, Type, Price, Method, SellerG, Distance
# Postcode, Bedroom2, Bathroom, Car, Landsize, BuildingArea, YearBuilt,
# PropertyCount
# Categorical
# Address->Street, Type, Method, SellerG

numeric_cols = ["Rooms", "Distance", "Postcode", "Bedroom2", "Bathroom", "Car", "Landsize", "BuildingArea", "YearBuilt", "Propertycount"]
cat_cols = ["Suburb", "Address", "Type", "Method", "SellerG"]

def get_xs(data):
    cols = cat_cols + numeric_cols
    x = data[cols]
    x["Address"] = x["Address"].str.split(" ").str.get(1).str.strip()
    return x

pd.options.mode.chained_assignment = None
data = pd.read_csv("melb_data.csv", index_col="Unnamed: 0")
x = get_xs(data)
y = data["Price"]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=196)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, cat_cols)
])

pipeline = Pipeline(steps=[
    ('preprocess', preprocess),
    ('xgb', XGBRegressor(objective='reg:squarederror', max_depth=7, n_estimators=250))
])

pipeline.fit(x_train, y_train)
print(r2_score(y_test, pipeline.predict(x_test)))