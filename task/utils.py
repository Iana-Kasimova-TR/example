import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import missingno as msno
from sklearn.impute import SimpleImputer, KNNImputer
from scipy.stats import chi2_contingency
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
import plotly.io as pio
import shap
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import xgboost as xgb

#!pip install yellowbrick
from yellowbrick.target import FeatureCorrelation
from yellowbrick.classifier import ClassBalance, ClassificationReport, ConfusionMatrix, DiscriminationThreshold
from yellowbrick.features import JointPlotVisualizer, pca_decomposition, RadViz, Rank1D, Rank2D

def display_target_balance(df, name_of_target, labels):
    # Separate the target and features as separate dataframes for sklearn APIs
    y = df[[name_of_target]].astype('int')


    # Specify the design matrix and the target vector for yellowbrick as arrays
    target_vector = y.values.flatten()
    # Target balance
    target_balance = ClassBalance(labels=labels)
    target_balance.fit(target_vector)
    target_balance.show();

def pca_decompose(df, features, target, classes):
    df_full = df.dropna().reset_index()
    X_sample = df_full[features].sample(200)
    y_sample = df_full[target].astype('int').iloc[X_sample.index.values].reset_index(drop=True)
    pca_decomposition(
    X_sample, y_sample, scale=True, classes=classes, projection=3
)


def display_feature_corr(df, target, num_features): 
    df = df.dropna()
    X = df[num_features]
    y = df[[target]].astype('int') 
    # Feature correlation (** requires dataframe and 1D target vector)
    feature_correlation = FeatureCorrelation(method='mutual_info-classification',
                                            feature_names=X.columns.tolist(), sort=True)
    feature_correlation.fit(X, y.values.flatten())
    feature_correlation.show();


def plot_bars_three_features(df, feature1, feature2, feature3, title):
    plt.figure(figsize=(16, 8))
    sns.barplot(x=feature1,
                y=feature2,
                hue=feature3,
                data=df)
    plt.title(title, fontsize=16)
    plt.xlabel(feature1, fontsize=16)
    plt.xticks(rotation=45)
    plt.ylabel(feature2, fontsize=16)
    plt.legend(loc="upper left")
    plt.show()


def plot_counts_in_time(df, groupby, feature_for_count, title):
    group_value = df.groupby(groupby)[feature_for_count].count()
    data = pd.DataFrame({groupby:list(group_value.index),
    feature_for_count:list(group_value.values)})
    plt.figure(figsize=(12, 8))
    sns.lineplot(x = groupby, y=feature_for_count, data=data, sizes=(2.5, 2.5))
    plt.title(title, fontsize=16)
    plt.xlabel(groupby, fontsize=16)
    plt.xticks(rotation=45)
    plt.ylabel("Number of guests", fontsize=16)
    plt.show()

def plot_proportion_category(df, feature_cat):
    fig, ax = plt.subplots()

    # Plot a normalized countplot
    df[feature_cat].value_counts(normalize=True).plot.barh()

    # Label
    ax.set(title='Proportion of Categories',
        xlabel='Proportion', ylabel='')

    plt.show();   


def get_importance_of_numerical_features(df, target_feature, num_features):
    num_features.append(target_feature)
    cancel_corr = df.corr()[target_feature]
    print(cancel_corr.abs().sort_values(ascending=False)[1:])

"""
Standartization for numerical, ordinal for categories, simpleimputer for missing values with different stratigies
The features are converted to ordinal integers.
This results in a single column of integers (0 to n_categories - 1) per feature.
"""
def provide_column_transformer(category_features, num_features):
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean", fill_value="Unknown")),
        ("scaler", StandardScaler())])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoding", OrdinalEncoder())])

    preprocessor = ColumnTransformer(transformers=[("cat", cat_transformer, category_features),
        ("num", num_transformer, num_features)])    
    return preprocessor   

    
def run_models(base_models, category_features, num_features, X, y, scoring, kfolds):
    preprocessor = provide_column_transformer(category_features, num_features)
    round = 4
    split = KFold(n_splits=kfolds, shuffle=True, random_state=42)
    min_scores = {}
    max_scores = {}
    mean_scores = {}
    std_dev = {}
    for name, model in base_models:
        # pack preprocessing of data and the model in a pipeline:
        model_steps = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', model)])
        cv_results = cross_val_score(model_steps, 
                                    X, y, 
                                    cv=split,
                                    scoring=scoring,
                                    n_jobs=-1)
        mean_scores[name] = np.round(np.mean(cv_results), round)
        max_scores[name] = np.round(np.max(cv_results), round)
        min_scores[name] = np.round(np.min(cv_results), round)
        std_dev[name] = np.round(np.std(cv_results), round)   
        print(f"{name} cross validation score: {mean_scores[name]} +/- {std_dev[name]} (std) min: {min_scores[name]}, max: {max_scores[name]}")
    scores = {"models": mean_scores.keys(), "scores": mean_scores.values()}
    fig = px.bar(scores, x='models', y='scores')
    fig.show()  
    return mean_scores, max_scores, min_scores, std_dev

"""
HYPEROPT or just gridsearcv and stratified k-fold
https://hyperopt.github.io/hyperopt/?source=post_page
"""
def tune_model(X, y, model, parameters, scoring):
    #X = df.loc[:, df.columns != 'is_canceled']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = GridSearchCV(model, parameters, n_jobs=5, 
                   cv=StratifiedKFold(n_splits=5, shuffle=True), 
                   scoring='roc_auc',
                   verbose=2, refit=True)

    clf.fit(X_train, y_train)           

    best_parameters, score = clf.best_params_, clf.best_score_
    print('Score:', score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))

    #test_probs = clf.predict_proba(X_test)[:,1]

    # run the grid search
    #optimize_hparams.fit(X, y)

    # fit the preprocessor - but we have it before 
    #X_encoded = optimize_hparams.best_estimator_['preprocessor'].fit_transform(X)
    #replace X on X_encoded

    # fit the model 
    #best_model = optimize_hparams.best_estimator_['model'].fit(X_encoded, y)
    best_model = clf.best_estimator_

    # calculate the Shap values
    shap_values = shap.TreeExplainer(best_model).shap_values(X)

    # plot the Shap values
    shap.summary_plot(shap_values, X, plot_type='bar')


def plot_missing_values(df, target):
    msno.matrix(df.sort_values(target))  


def plot_unique_values(df, column_names):
    for name in column_names:
        print(df[name].value_counts())


def cramers_V(var1,var2) :
    crosstab =np.array(pd.crosstab(var1,var2, rownames=None, colnames=None)) # Cross table building
    stat = chi2_contingency(crosstab)[0] # Keeping of the test statistic of the Chi2 test
    obs = np.sum(crosstab) # Number of observations
    mini = min(crosstab.shape)-1 # Take the minimum value between the columns and the rows of the cross table
    return (stat/(obs*mini))

def corr_mtx_cat_features(df):
    rows = []
    for var1 in df.columns:
        col = []
        for var2 in df.columns:
            cramers =cramers_V(df[var1], df[var2]) # Cramer's V test
            col.append(round(cramers,2)) # Keeping of the rounded value of the Cramer's V  
        rows.append(col)
  
    cramers_results = np.array(rows)
    return pd.DataFrame(cramers_results, columns = df.columns, index = df.columns)    


   