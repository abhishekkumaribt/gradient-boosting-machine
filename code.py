# --------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report , accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")



# Explore the data 
df = pd.read_csv(path)

# mean and standard deviation of their age


# Display the statistics of age for each gender of all the races (race feature).
m_f = df.sex.value_counts()
fig = m_f.plot(kind="bar", figsize=[5,5], color="green")
for x,y in enumerate(m_f):
    fig.text(x-0.05 , y*0.4, y, rotation=90, color="yellow", fontsize=18)

german_per = df[df["native-country"]=="Germany"].shape[0]/df.shape[0]
print(german_per)

sal50K = df[df["salary"]==">50K"]
print(sal50K["age"].mean(), sal50K["age"].std())

# encoding the categorical features.
Amer_Indian_Eskimo_Male_max_age = df.groupby(["race","sex"])[["age"]].describe().loc["Amer-Indian-Eskimo","Male"]["age"]["max"]
print(Amer_Indian_Eskimo_Male_max_age)

# Split the data and apply decision tree classifier
df.salary = df["salary"].replace({"<=50K":1, ">50K":0})
df = df.replace("?", np.nan)
df = df.dropna()
for col in df.select_dtypes('object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop(columns="salary")
y = df.salary
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# Perform the boosting task
dt = DecisionTreeClassifier()
dt.fit(X_train_val, y_train_val)
print(dt.score(X_test_val, y_test_val))

vot_cla = VotingClassifier([("Decision Tree", DecisionTreeClassifier()), ("Random Forest", RandomForestClassifier(max_depth=6, n_estimators=50)), ("Random Forest 2", RandomForestClassifier(max_depth=6, n_estimators=50)), ("Logistic Regression", LogisticRegression())], voting='soft')
vot_cla.fit(X_train_val, y_train_val)
print(vot_cla.score(X_test_val, y_test_val))

#  plot a bar plot of the model's top 10 features with it's feature importance score
gboost = GradientBoostingClassifier(n_estimators=100, max_depth=6)
gboost.fit(X_train_val, y_train_val)
print(gboost.score(X_test_val, y_test_val))

imp_fea = pd.DataFrame({"feature":X_train.columns, "score":gboost.feature_importances_})
imp_fea = imp_fea.sort_values("score", ascending=False).head(10)
imp_fea.index = imp_fea.feature
imp_fea.drop(columns="feature", inplace=True)
imp_fea.plot(kind="bar")

#  Plot the training and testing error vs. number of trees
gboost_10 = GradientBoostingClassifier(n_estimators=10, max_depth=6)
gboost_50 = GradientBoostingClassifier(n_estimators=50, max_depth=6)
gboost_100 = GradientBoostingClassifier(n_estimators=100, max_depth=6)

gboost_10.fit(X_train_val, y_train_val)
train_err_10 = 1-(gboost_10.score(X_train_val, y_train_val))
gboost_50.fit(X_train_val, y_train_val)
train_err_50 = 1-(gboost_50.score(X_train_val, y_train_val))
gboost_100.fit(X_train_val, y_train_val)
train_err_100 = 1-(gboost_100.score(X_train_val, y_train_val))
training_errors = [train_err_10, train_err_50, train_err_100]
training_errors

validation_err_10 = 1-(gboost_10.score(X_test_val, y_test_val))
validation_err_50 = 1-(gboost_50.score(X_test_val, y_test_val))
validation_err_100 = 1-(gboost_100.score(X_test_val, y_test_val))
validation_errors = [validation_err_10, validation_err_50,validation_err_100]
validation_errors

testing_err_10 = 1-(gboost_10.score(X_test, y_test))
testing_err_50 = 1-(gboost_50.score(X_test, y_test))
testing_err_100 = 1-(gboost_100.score(X_test, y_test))
testing_errors = [testing_err_10, testing_err_50,testing_err_100]
testing_errors

pd.DataFrame([training_errors, validation_errors, testing_errors], index=["training_errors", "validation_errors", "testing_errors"], columns=["gboost_10", "gboost_50", "gboost_100"]).T.plot(kind="bar")


