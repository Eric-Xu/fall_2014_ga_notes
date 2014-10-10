
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