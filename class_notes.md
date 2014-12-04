
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

  Bernoulli Distribution
    - Each trial has two possible outcomes, in the language of reliability called success and failure.
    - The trials are independent. Intuitively, the outcome of one trial has no influence over the outcome of another trial.
    - On each trial, the probability of success is p and the probability of failure is 1−p

  Binomial Distribution
    - Binomial distribution is a sum of independent and evenly distributed Bernoulli trials.
    - Each trial has two possible outcomes, in the language of reliability called success and failure.
    - The trials are independent
    - x is a discrete number representing the number of successes in n trials
    - specific order of win-lose trial outcomes does not matter; just the number of times it happened

  Multinomial Distribution
    - a generalization of the binomial distribution
    - Each trial has more than two possible outcomes
    - the outcomes are mutually exclusive and exhaustive (at least one will occur)
    - the probability of each outcome added together sums to 1
    - the probabilities are also constant between trials
    - "In a random sample of 10 Americans, what is the probability 6 have blood type O, 2 have type A, 1 has type B, and 1 has type AB?"
    - "An urn containing 8 red balls, 3 yellow balls, and 9 white balls. 6 balls are randomly selected WITH replacement."
    - https://www.youtube.com/user/jbstatistics/videos

  Multivariate Hypergeometric Distribution
    - "An urn containing 8 red balls, 3 yellow balls, and 9 white balls. 6 balls are randomly selected WITHOUT replacement."


### Oct 27:
  AUC
    - a good way to measure the accuracy of your classifier
  Confusion Matrix
    -

### Oct 29:
  term project presentations
  Data Explor06
    - one row per concert (?)
    - naive bayes
    - kfolds doesn't do a good job randomizing the training and test data for time series data; it systematically picks rows in consecutive blocks; therefore we want to randomize the df
    - Goal: predict what year a concert happened based on the set list

### Nov 3:
  Time series
    - cannot use cross validation for time series data because data is dependent
    - can only apply time series analysis on stationary data (mean of dataset is zero)
    - for non-stationary data we first need to do a transformation
    - brownian motion
      - high auto correlation
      - mean can be at any level
    - white noise is just random data
      - low auto correlation
      - mean is zero
    - brownian motion and white noise are at two extremes of how time series data could look like. jarret is using these as ways to demonstrate how to use correlations to pick the number of covariates

  Cross validating time series
    - kfolds does not work for time series because we need our training and test data to be in random consecutive batches
    - instead
      - we do this manually in 11_timeseries/cross_validating_time_dependent_models
      - randomly select a batch of consecutive data
      - try to predict the next batch

### Nov 5:
  Guest speaker
    - Carmelo

  Kaggle Africa soil winner
    - step 1: preprocessing
      - standard deviation
      - log transform
    - step 2: models
    - step 3: metrics

### Nov 10:
  Clustering
    - hard clustering do not overlap
    - soft clustering can overlap;
    - can be used as a preprocessing step before applying categorization analysis

  K-means
    - what's the best way to choose k?
    - k centroids; reached optimal solution when centroids stop moving
    - jaccard coefficient is a popular metric for text mining problems
    - fast so good for large datasets
    - disadvantages: local minima problem; how many k's?

  Elbow method to picking k
    - inertia on the y-axis

### Nov 12:
  Sushan
    - bayesian analysis
    - python nltk tagging; nltk's key word extraction
    - nonstandard distance formula
      - normalized google distance) to compare keyword list
      - cosine similarity for document similarities; nltk.cluster.kmeans
    - analyize the body as well?
    - nytimes tags api

  Recommendation Systems
    - Rob works on article recommendation
    - linear algebra problem; not clustering
    - Content based filtering
      - map users to feature space
      - disadvantage is must map each item into a feature space; interns or Mechanical Turk
      - hard to create cross content recommendations; Pandora cannot recommend movies
    - Collaborative filtering
      - when two people share interests; they will like similar things
      - how users will rate items they have not used
      - does not scale well; item-item similarity matrix
      - currently the buzz word for recommendation systems; if interviewing for a job that builds rec systems, be prepared to talk about adv/disadv of CF


### Nov 12:
  NLP
    - Peter Norvig coursera AI
    - corpus is a structured set of texts; the Brown Corpus
    - filtering stop words:
      - remove the most frequent words; Tipf's law
      - remove anything with 3 letters or less
      - define a list of stop words
    - TF-IDF is larger for words that occur more in a single document but less in all docs
    - cosine similarity is most common measure for doc similarity
    - cosine similarity of 90 deg means two articles are orthogonal


### Nov 19:
  A|B testing
    - Bayes Theorem
      - P(B|A) likelihood
      - P(A) prior
      - P(B) normalization constant
      - P(A|B) posterior
    - Headline testing from Rob's work
      - frequentist probability measures a proportion of outcomes
      - bayesian measures degree of belief; ctr is within a certain range
      - frequentist approach flaws:
        - need define number of page views (ie 20,000) to test p-value;
        - different sites will hit those numbers at different rates
        - some will hit 20,000 in 2 secs; others will take 2 days
      - Bayesian approach
        - Anscombe boundary; stop a test as soon as a conclusion has been reached
        - beta distribution is good for binary outcomes
          - probability of probabilities
          - model baseball batting average; hits vs misses
    - additional reading
      - multi arm bandit

  Data Explor09
    - email: jarret.petrillo@gmail.com
    - subject: [Dataexplor01]


### Nov 24:
  Decision Trees
    - ID3 is a greedy search algorithm
    - greedy algorithms uses the most optimal decision at that point in time;
    - weakness is that the optimal point might come later; found the local optimum, not the global optimum; prone to over-fitting
    - advantages: features do not need to be scaled or mean-centered
    - never used in the real world. people prefer random forests and ensemble methods

  Random Forest
    - solves the greediness of decision trees; avoids finding the local optimum

  Emsemble Learning
    - bagging vs boosting


### Dec 1:
  HDFS
    - Hadoop won't be around in 5 years
    - new technology built on top of Hadoop like Apache Spark

  Parquet columnar data structure
    - column oriented makes it easier to select only relevant columns; group values within columns

  Geospatial
    - Blake Shaw of Foursquare "Data Driven NYC 20" video
    - Foursquare problem is a combination of NLP, recommendation system, geospatial

  Prep work for weds: (bring personal laptop)
    - http://hortonworks.com/products/hortonworks-sandbox/#install


### Dec 3:
  Big Data
    - volume, velocity, variety
    - moving code to the data

  Map Reduce (framework)
    - the term "MapReduce" comes from functional programming
    - Hadoop is an open-source java implementation of the map-reduce framework

  Storm (Varun Vijayaraghavan)
    - real time stream processing framework
    - continuous data can come at a high velocity (sensory data 100,000 points/sec)
    1. data source passes into backend server (data ingestion)
    2. splits data into data streams for analysis

  Apache Spark
    - batch based data parallel system; general purpose cluster computing system

  Hortonworks Sandbox
    - localhost:8888
    - localhost:8000
    - Query Editor:
      > show table;
      > describe sample_07;
      > select * from sample_07 limit 5;
      > select count(*) from sample_07;
      > select * from sample_07 join sample_08 on sample_07.code = sample_08.code;
    - Rob: learn Hive and Pig

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
# GridSearchCV # hyperparameter grid search to find best model parameters
http://nbviewer.ipython.org/github/herrfz/dataanalysis/blob/master/assignment2/samsung_data_prediction_submitted.ipynb

evan will thomas granger liz
