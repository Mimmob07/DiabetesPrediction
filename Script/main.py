import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

warnings.filterwarnings("ignore")
diabetes = pd.read_csv("../diabetes.csv")
diabetes.iloc[:, 1:7] = diabetes.iloc[:, 1:7].replace(0, np.NaN)
features=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',"Age","BMI","DiabetesPedigreeFunction"]
filledRows = diabetes.dropna()
diabetesPredicted = diabetes.copy()
accScores = []
f10 = []
f11 = []
re0 = []
re1 = []
pr0 = []
pr1 = []


for col in diabetes.columns:
    tmpImputer = SimpleImputer(missing_values=np.nan, strategy="mean")

    if col == "Pregnancies" or col == "DiabetesPedigreeFunction" or col == "Age" or col == "Outcome":
        continue
    nanRows = diabetes[diabetes[col].isna()]
     
    filledRowsX = filledRows.drop(columns=[col])
    filledRowsY = filledRows[[col]]
    nanRowsX = nanRows.drop(columns=[col])
    nanRowsY = nanRows[[col]]
    for icol in nanRowsX.columns:
        nanRowsX[icol].fillna(diabetes[icol].mean(), inplace=True)
        
    linRegModel = LinearRegression()
    linRegModel.fit(filledRowsX, filledRowsY)
    
    linRegPred = linRegModel.predict(nanRowsX)
    diabetesPredicted.loc[diabetesPredicted[col].isna(), col] = linRegPred

kfold = KFold(n_splits=5, shuffle=True, random_state=20)
for train, test in kfold.split(diabetesPredicted):
    train_x = diabetesPredicted.iloc[train, :8]
    train_y = diabetesPredicted.iloc[train, 8]
    test_x = diabetesPredicted.iloc[test, :8]
    test_y = diabetesPredicted.iloc[test, 8]
    linmodel = LinearRegression()
    linmodel.fit(train_x, train_y)
    linear_pred = (linmodel.predict(test_x) > 0.55) * 1
    linmodel_fi = permutation_importance(linmodel, train_x, train_y)
    accScores.append(accuracy_score(test_y, linear_pred))
    f1sc = f1_score(test_y, linear_pred, average=None)
    resc = recall_score(test_y, linear_pred, average=None)
    prsc = precision_score(test_y, linear_pred, average=None)
    f10.append(f1sc[0])
    f11.append(f1sc[1])
    re0.append(resc[0])
    re1.append(resc[1])
    pr0.append(prsc[0])
    pr1.append(prsc[1])

tmp = pd.DataFrame({'Feature': features, 'Feature importance': abs(linmodel_fi['importances_mean'])})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
fig = px.bar(tmp,x='Feature',y='Feature importance',color='Feature importance',title="Features Importance of Linear Regression Model",
            labels=dict(x="Feature",y="Feature importance",color="Feature importance"),color_continuous_midpoint=0.8,
            width=600,height=600,template="plotly_dark")
fig.show()

cmLin = confusion_matrix(test_y, linear_pred)
dispLin = ConfusionMatrixDisplay(confusion_matrix=cmLin)
dispLin.plot()
plt.show()