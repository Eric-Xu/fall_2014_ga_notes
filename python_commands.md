# General Python

## Type casting
```python
print float("99")
print int(2.5)
print str(99)
```

## Try/Catch
```python
try:
  # code
except Exception as err:
  print(err)
  continue # This tells the computer to move on to the next item after it encounters an error
```

# Pandas

## Import data:
  ```python
  orig_data = pd.read_csv("africa_ph.csv")
  data = orig_data.copy()

  pd.read_csv(filename, sep=" ", index_col=0)
  ```

## Common data prep steps:
```python
df.dropna(inplace=True)

df["League"] = pd.factorize(df["League"])[0]

df.replace(to_replace='NN', value=-1, inplace=True)

predictors = [col for col in df.columns if col != "Salary"]
X = df[predictors].values
y = df.Salary.values
```

## Series transformations:
```python
heart_data['sex']=heart_data['sex'].apply(str)

data.Title = data.Title.apply(lambda x: x.split(' (')[0])

new_df = df[['US10Y_yield','US10Y_date']]
new_df = new_df.dropna()
new_df.date = new_df.date.apply(pd.to_datetime)
new_df.set_index('date', inplace=True)
new_df = new_df.drop('US10Y_date', axis=1)
```

## Create a new categorical column:
```python
# fall_2014_assignments/lab04/EDA_lab.ipynb
def age_grouping(age):
  if age < 18:
    return "<18"
  elif age >= 18 and age < 65:
    return "18-64"
  elif age > 65:
    return "65+"
  else:
    return np.nan

df['age_group'] = df.Age.apply(lambda x: age_grouping(x))

# fall_2014_lessons/06_dimensionality_reduction/lab/regularization2.ipynb
test = pd.DataFrame({'x': [1, 1, 2, 2, 1, 3], 'y':[1, 2, 2, 2, 2, 1]})
pd.factorize(test['x'])
# => (array([0, 0, 1, 1, 0, 2]), Int64Index([1, 2, 3], dtype='int64'))
```

## Create a new dataframe:
```python
# fall_2014_lessons/03_linear_regression/Working\ With\ Data.ipynb
new_df = pd.DataFrame(index=range(20), columns=["A","B"])
new_df['A'].fillna("No!", inplace=True)
new_df['B'].fillna("Yes!", inplace=True)

predictions = pd.DataFrame(columns=["Actual", "Predicted"])
predictions.Actual = x_vals
predictions.Predicted = y_vals
```

## Convert Data into a Standardized Normal Distribution:
```python
# fall_2014_assignments/dataexplor02/exu/africa_mark_holt.ipynb
snd_data = (data - data.mean())/data.std()
type(snd_data) # => pandas.core.frame.DataFrame
```

## Modeling with Standardized Normal Distribution and Z-scores:
```python
# assuming model has already been fitted where Infrared02 is the predictor (x) and pH is the response (y)
# model = LinearRegression(fit_intercept=True)
# model.fit(x, y)

infrared02_val = 0.04079
infrared02_z_score = (infrared02_val - data["Infrared02"].mean()) / data["Infrared02"].std()

pH_z_score = model.predict(z_score)
predicted_pH = (pH_z_score * data["pH"].std()) + data["pH"].mean()
```

## Finding Coorelations:
```python
data.corr()
data.corrwith(data.pH)

ordered_corrs = fabs(snd_data.corrwith(snd_data.pH)).order(ascending=False)
ordered_corrs.plot()
```

## Preparing a Test and Training Dataset:
```python
# fall_2014_assignments/dataexplor02/exu/africa_mark_holt_exu.ipynb
boolean_filter = np.random.random_integers(0,1,len(snd_data))

train = snd_data[boolean_filter==1].copy()
test = snd_data[boolean_filter==0].copy()

# http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/machine_learning/scikit-learn/scikit-pipeline.ipynb
# we split our dataset into two subsets: A training dataset (60% of the samples) and a test dataset (40% of the samples from the original dataset).
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=12345)
```

## Computing the Mean Squared Error (MSE):
```python
# fall_2014_assignments/dataexplor02/exu/kaggle_africa_mholt_exu.ipynb
predictions = pd.DataFrame(columns=["Actual","Predicted"])
predictions.Actual = test_actual_ph
predictions.Predicted = test_predicted_ph

predictions.corrwith(predictions.Actual)

predictions.plot(figsize=(18,4))

# use sklearn's bult in mse
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(predictions.Predicted, predictions.Actual)

# or calculate the mse manually
errors = predictions.Predicted - predictions.Actual
squared_errors = errors**2
mse = squared_errors.sum()/squared_errors.count()
```


# Matplotlib

## Default configurations:
```python
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.mpl_style', 'default')
```

## Inline configurations:
```python
plt.figure(figsize=(27,9))
plt.xlabel('mid-infrared', fontsize=24)

_c = [random(), random(), random(), .2]
plt.plot(fun, work, color=_c, alpha=1)
```

## Subplots:
```python
data[["pH","REF2","BSAN"]].plot(subplots=True, figsize=(15,5))

http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/machine_learning/scikit-learn/scikit-pipeline.ipynb
```


# Scikit Learn

## Fit a simple y=ax + b model:
```python
# fall_2014_assignments/dataexplor02/exu/kaggle_africa_ph.ipynb
x = data.Infrared02.reshape(len(data.Infrared02),1) # predictor
y = data.pH # response

model = LinearRegression(fit_intercept=True)
model.fit(x, y)
print ("Model coefficient: %.5f, and intercept: %.5f" % (model.coef_, model.intercept_))

predicted_pH = model.predict(0.04079)
# or calculate predicted_pH manually
# predicted_pH = model.coef_(0.04079) + model.intercept_
```

## Plot the linear regression line:
```python
# pick 100 hundred points equally spaced from the min to the max
x_vals_for_reg_line = np.linspace(data.Infrared02.min(), data.Infrared02.max(), 100).reshape(100,1)
y_vals_for_reg_line = model.predict(x_test)

plt.plot(x_vals_for_reg_line, y_vals_for_reg_line)
```

## Computing Cross Validation Metrics:
```python
from sklearn.cross_validation import cross_val_score

predictors = snd_data.Infrared02.reshape(len(snd_data.Infrared02), 1)
# predictors = snd_data[['ELEV', 'REF2', 'BSAN']]

accuracy_score = cross_val_score(model, predictors, snd_data.pH, n_jobs=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (accuracy_score.mean(), accuracy_score.std() * 2))

mse_score = cross_val_score(model, predictors, snd_data.pH, n_jobs=5, scoring="mean_squared_error")
print("MSE: %0.2f (+/- %0.2f)" % (mse_score.mean(), mse_score.std() * 2))
```

## K-fold Cross Validation:
"The Right Way to Cross Validate" - http://nbviewer.ipython.org/github/cs109/content/blob/master/lec_10_cross_val.ipynb

```python
# fall_2014_assignments/dataexplor02/exu/kaggle_africa_mholt_exu.ipynb
from sklearn.cross_validation import KFold
from random import random

inf02_z_scores = snd_data.Infrared02.reshape(len(snd_data.Infrared02),1)
ph_z_scores = snd_data.pH.values

kfold = KFold(len(ph_z_scores), n_folds=3)
mses = []

for train, test in kfold:
    Xtrain, ytrain, Xtest, ytest = inf02_z_scores[train], ph_z_scores[train], inf02_z_scores[test], ph_z_scores[test]

    model = LinearRegression(fit_intercept=True)
    model.fit(inf02_z_scores, ph_z_scores)

    ypred = model.predict(Xtest)
    mses.append(mean_squared_error(ypred, ytest))
    # mses.append(model.score(ypred, ytest))

    plt.figure(figsize=(9,9))
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    _c = [random(), random(), random(), .5]
    plt.plot(ypred, ytest, 'o', color=_c)
    plt.plot(ytest, ytest, '-')

print mses
print("CV mean squared error is ", np.mean(mses))
```
