
- The dependent (or output) variable is the price of the house that we're trying to predict.
- The predictor (or input) variable is the variable used to guide our prediction.
- The p-value for each term tests the null hypothesis that the coefficient is equal to zero (no effect). A low p-value (< 0.05) indicates that you can reject the null hypothesis. Conversely, a larger (insignificant) p-value suggests that changes in the predictor are not associated with changes in the response.
- Multivariate regression analysis is when multiple independent variables are hypothesized to be associated with a single dependent variable.
- Multicollinearity (also collinearity) is a statistical phenomenon in which two or more predictor variables in a multiple regression model are highly correlated, meaning that one can be linearly predicted from the others with a non-trivial degree of accuracy.

How to use Pairwise and Partial Correlations to simplify a multivariate regression:
  http://www.jmp.com/academic/pdf/cases/12HousingPrices.pdf

### Sep 29:
  Agenda
    - intro to machine learning
    - intro to python tools

  - supervised algorithm is making predictions; clear understanding of problem at hand
    - dependent variables; predictor variable
    - can be categorized as regression, classification, and sometimes clustering
    - examples include spam filtering; character recognition
    - document clustering can be either or; evergreen means only certain file types are allowed
  - unsupervised learning is
    - trying to find hidden structure in unlabeled data
    - no outcome variable, just a collection of inputs; hard to tell if you are doing well
    - the most common unsupervised learning method is cluster analysis
      - Hierarchical clustering; k-Means clustering; self-organizing maps; hidden Markov models
    - used in bioinformatics for sequence analysis and genetic clustering; in medical imaging for image segmentation; and in computer vision for object recognition.
  - Rob recommends "Python for Data Analysis" for learning about pandas; a bit out dated
  - numpy
      - `heart_data['sex']=heart_data['sex'].apply(str)`
      - apply() mimics map() from functional programming; takes advantage of underlying C's speed which you don't get from writing a Python for-loop.

  - SUPERVISED; UNSUPERVISED; REGRESSION; CLASSIFICATION; CLUSTERING

  - How to config ipython notebook's load dir inside of the data-science-dst virtual box
    ```sh
    vagrant ssh
    subl ~/.ipython/profile_dst/ipython_notebook_config.py
    c.FileNotebookManager.notebook_dir = u'/ga_fall_2014'
    c.NotebookManager.notebook_dir = u'/ga_fall_2014'
    ```

  Question: why is count() returning different values?
    ```python
    data.Ozone.isnull().count() # => 153
    data.Ozone.isnull().sum() # => 37
    data.Ozone.count # => 116
    ```

### Oct 1:
  Linear regressions
    - time series data is almost always non-independent, meaning one outcome influences the next
    - ordinary least squares(OSL) method
    - http://dss.princeton.edu/online_help/analysis/interpreting_regression.htm
    - http://www.datarobot.com/blog/multiple-regression-using-statsmodels/

  Kaggle competition: Africa Soil Property Prediction Challenge
    - step 1 is spacial reduction; there are over 3000 columns!
    - send to all instructors; email subject: dataexplor02

  Python Stats Libraries:
    - Pandas (PD) is mainly a package to handle and operate directly on data.
    - Scikit-learn is doing machine learning with emphasis on predictive modeling with often large and sparse data
    - Statsmodels (SM) is doing "traditional" statistics and econometrics, with much stronger emphasis on parameter estimation and (statistical) testing.

  - HEDONIC, SUPPORT VECTOR MACHINES (SVM)

### Oct 6:
  Data visualization
    - Edward Tufte
    - tools: wrappers on top of matplotlib include seaborn, vincent, and bokeh
  Exploratory data analysis (EDA)
    - Outbrain's jobs queue example:
      - graph it; rescale; apply log transformations; filter on subset; pull in more data

### Oct 8:
  Agenda
    - cross validation
    - model selection criteria

  Data Exploration02 discussion
    - Louisa did a Pearson correlation
    - pH data in Jarret's file was modified
    - skewness shows how far to the left and right
    - kurtosis means how pointy is the data (?)
    - Mark subsetted the data based on elevation; then reran the correlations on that subset and decided to run a regression based on Ref2, Infrared02, and BSAN

  Model Selection
    - mean squared error is one way to measure model accuracy
    - but minimizing the MSE might lead to overfitting
    - balance between MSE and degrees of freedom
    - the following has 2 degrees of freedom:
      `y = B0 + B1x + B2x2`
    - StatsModel is newer; a Python port of R
    - StatsModel summary:
        F-stats bigger than 2 is good
        R-squared close to 1 is good
    - machine learning evolved out of CS dept; scikit-learn
    - statistical learning evolved out of math dept; R; StatsModel

  Multiple Linear Regression
    - the ideal regression coef scenario is when the predictors are uncorrelated

  Cross Validation
    - K-fold is most common
    - also a 'Leave one out'
    - use cross validation when your dataset isn't large enough to be split into a test set

  Rob's day to day model validation (depends on the type of data)
    - F-statistic, AIC, its hard to say
    - that's because most are time series data without many predictors

  Jarret's approach
    - first check P value and eliminate predictors
    - then check AIC, Omnibus, Kurtosis

  Data Exploration03
    - raw Bloomberg data; clean it
    - Question: How do you interpret a regression model? How does a change in inputs affect the output?
    - LIBOR - OIS = good spread


### Oct 15:
  Agenda
    - recap subset selection
    - regularization
    - dimension reduction

  Regularization
    - RSS is residual sum of squares; try to minimize this
    - lasso & ridge regressions (Rob almost never uses in day job; but good to know)

  Multicollinearity
    - high correlations in data; therefore model suffers from multicollinearity
    - PCA can be used to solve this; it is not a model; it is a preprocessing step for variables to be used in a model

  L1/Lasso & L2/Ridge
    - overfitting; robustness

  Heteroscedasticity

  Using PCA to Summarize High Dimensional Data
    - a way to summarize covariance in data

  Partition Training/Testing Datasets
    - typically 80/20

  K-fold
    - root of mse, absolute mse, explained

  Data Explor03
    - df.merge (ask Jarret about how the cleaned data should look like)


### Oct 20:
  Agenda
    - logit regression
    -  intro to grid search

  Data Explor04 walkthrough
    - looking at mean and standard deviation; here all of the std's are the same, indicating that the data has been normalized
    - y here has not been transformed
    - definitely do a histogram of the dependant variable
    - pd.scatter_matrix; use this when there are less than 10 predictors; otherwise it will choke up the kernel

    1. df.describe
    2. plot histogram for each var
    3. df.head
    4. df.scatter_matrix
    5. df.corr
    6. model with all predictors as baseline
    7. use cross validation to select subset (SelectKBest is blackbest; python's itertools is more transparent)
    8. compare baseline with lasso and ridge w/subset or all predictors
    9. develop higher order polynomial model

  Classification
    - classification you know the groups in advance (supervised)
    - clustering is unsupervised
    - confusion matrix
    - accuracy is fraction of instances predicted correctly
    - precision is fraction of POSITIVES that are indeed POSITIVES
    - recall
    - fall-out is the false positive rate

  Logistic Regression
    - modeling for classification data
    - can linear regression also be used for classification? Yes, if outcomes fall within (?)
    - linear regression fails because it can produce results where probability is greater than 1 (100%) or less than 0 (0%)
    - linreg also fails because categories are arbitrarily assigned so doesn't have to be


### Oct 22:
  Agenda
    - probability theory
    - bayes theorem
    - naive bayes classifier

  Jarret's iterative search algorithm
    - iteratively select best features; so now instead of having 2^n combinations, it now becomes 2*n combinations
    - also called 'forward step-wise'

  Probability
    - between 0 and 1
    - joint probability P(AB)
    - conditional probability P(A|B)

  Bayes Theorem
    - frequentist vs bayesian

  Naive Bayes
    - given an article, assume that 'ebola' and 'virus' are conditionally non-independent

  Data Explor05 - sentiment analysis of tweets
    - "@DataDAVE thanks for the awesome twitter dataset!!"
    - "I just don't understand lasso!??!"


### Question for Rob
```python
# when to use cv and n_jobs?
# http://scikit-learn.org/stable/modules/cross_validation.html
model1 = LinearRegression(fit_intercept=True)
cv_score     = cross_val_score(model1, df[['ELEV', 'REF2', 'BSAN']], df['pH'], cv=5).mean()
n_jobs_score = cross_val_score(model1, df[['ELEV', 'REF2', 'BSAN']], df['pH'], n_jobs=5).mean()
print cv_score, n_jobs_score # => -0.0694517866041 -0.0446749474373

# which metric to look at?
accuracy_score = cross_val_score(model1, df[['ELEV', 'REF2', 'BSAN']], df['pH'], n_jobs=5).mean()
mse_score = cross_val_score(model, predictors, snd_data.pH, n_jobs=5, scoring="mean_squared_error").mean()

# how to choose inputs for K-fold?
kfold = KFold(len(snd_data), n_folds=10)
```

1. what's your project?
2. where's the data coming from?
3. objectives: what you're trying to learn and predict?

# GridSearchCV # hyperparameter grid search to find best model parameters
http://nbviewer.ipython.org/github/herrfz/dataanalysis/blob/master/assignment2/samsung_data_prediction_submitted.ipynb

evan will thomas

Sunday office hour 3:00-4:45pm