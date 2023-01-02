import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('car_data3.csv')

# Dropping NA and Duplicates entries 
df = df.dropna()
# dropna(): removes missing data

df = df.drop_duplicates()
# drop_duplicates(): removes duplicate rows based on all columns.

# Removing Unnecessary Columns
df = df[df["owner"] != "Test Drive Car"]

df.reset_index(drop=True, inplace=True) #reset index

# Getting Company Name from Name of the Car
name = df["name"]
names = []

for i in range(len(name)):
    c = str(name[i]).split(" ")[0]
    names.append(c)
    
df["company"] = pd.DataFrame(names)
df.drop(['name'], axis=1, inplace=True)

cat_cols = ["fuel", "seller_type", "transmission", "owner", "company", "torque"]
le=preprocessing.LabelEncoder()
df[cat_cols]=df[cat_cols].apply(le.fit_transform)

df.mileage = df.mileage.apply(lambda x: float(x.replace("kmpl", "")))
df.engine = df.engine.apply(lambda y: float(y.replace("CC", "")))
df.max_power = df.max_power.apply(lambda z: float(z.replace("bhp", "")))

X = df.drop(columns=["selling_price"])
y = df["selling_price"]
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 0.8, random_state=0)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)