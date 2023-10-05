'''
Source code for BMW thinktank interview

Juan Betancourt
juan.beta98@gmail.com
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,make_scorer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


################ Upload Data ################
AuditHistory = pd.read_csv('./Data/AuditHistory.csv')                   # 7690 entries
SupplierPerformance = pd.read_csv('./Data/SupplierPerformance.csv',index_col='SupplierId')     # 5823 entries

Suppliers = list(AuditHistory['SupplierId'].unique())               # 818 Suppliers
Derivatives = list(AuditHistory['DerivativeName'].unique())         # 75 Derivatives
ProductionLines = list(AuditHistory['ProductionLine'].unique())     # 4 Production Lines
Regions = list(AuditHistory['DerivativeRegion'].unique())           # 6 Regions



################ Ploting functions ################
def plot_categorical_distribution(col_name):
    if col_name != 'BadSupplierIndicator':
        counts = AuditHistory[col_name].value_counts()
    else:
        counts = SupplierPerformance[col_name].value_counts()

    # Setting Seaborn style
    sns.set(style="whitegrid")

    # Creating a bar plot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.index, y=counts, palette="viridis")

    # Adding labels and title
    plt.xlabel(col_name, fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title(f'Distribution of {col_name}', fontsize=16)

    # Rotating x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Displaying the plot
    plt.show()

def plot_numerical_distribution(col_name):
    if col_name != 'BadSupplierIndicator':
        # Setting Seaborn style
        sns.set(style="whitegrid")

        # Creating a histogram using Seaborn
        plt.figure(figsize=(10, 6))
        sns.histplot(SupplierPerformance[col_name], bins=20, color='skyblue', edgecolor='black')

        # Adding labels and title
        plt.xlabel(col_name, fontsize=14)
        plt.ylabel('Observations', fontsize=14)
        plt.title(f'Distribution of {col_name}', fontsize=16)

        # Displaying the plot
        plt.show()
    else:
        plot_categorical_distribution(col_name)

def code_categorical_columns(df, categorical_columns):
    encoding_dicts = {}
    for col in categorical_columns:
        if col[0]=='T' or col=='Result':
            df[col] = df[col].map({'-': 1, 'Green': 2, 'Yellow': 3, 'Red': 4})
        else:
            unique_categories = df[col].unique()
            encoding_dict = {category: code + 1 for code, category in enumerate(unique_categories)}
            encoding_dicts[col] = encoding_dict
            df[col] = df[col].map(encoding_dict)
        
    return df, encoding_dicts

def plot_audit_colors(df, SupplierId, DerivativeName):
    # Filter the DataFrame for the specified SupplierId and DerivativeName
    filtered_df = df[(df['SupplierId'] == SupplierId) & (df['DerivativeName'] == DerivativeName)]

    # Check if there is any data for the given filter
    if filtered_df.empty:
        print(f"No data found for SupplierId {SupplierId} and DerivativeName {DerivativeName}")
        return

    # Set the 'RecentMonth' column as the index for time series plotting
    filtered_df.set_index('RecentMonth', inplace=True)
    filtered_df = filtered_df.sort_index()
    # print(filtered_df)

    # Create a color map for mapping status to color
    color_map = {'-': 'white', 'Green': 'green', 'Yellow': 'yellow', 'Red': 'red'}

    # Plot the squares for each stage
    plt.figure(figsize=(12, 6))
    stages = ['T1Color', 'T2Color', 'T3Color']
    for stage in stages:
        plt.scatter(filtered_df.index, [stages.index(stage) + 1] * len(filtered_df), c=filtered_df[stage].map(color_map), marker='s', s=500, label=stage)

    # Customize the plot
    plt.title(f"Results per Audit Stage in Time")
    plt.xlabel('Date')
    plt.yticks([1, 2, 3], ['T1', 'T2', 'T3'])
    # plt.legend()

    # Show the plot
    plt.show()

def generate_temporal_WDandBL(AuditHistory,SupplierPerformance):
    WrongDeliverisInTrim,BacklogsInTrim = list(),list()

    for i in AuditHistory.index:
        if AuditHistory['SupplierId'][i] not in list(SupplierPerformance.index):
            WrongDeliverisInTrim.append(pd.NA)
            BacklogsInTrim.append(pd.NA)
            continue
        
        if AuditHistory['RecentMonth'][i][-2:] in [str(j) for j in ['08','09','10']]:
            WrongDeliverisInTrim.append(SupplierPerformance['Amount_WD_3M'][AuditHistory['SupplierId'][i]])
            BacklogsInTrim.append(SupplierPerformance['Amount_Backlogs_3M'][AuditHistory['SupplierId'][i]])
        elif AuditHistory['RecentMonth'][i][-2:] in [str(j) for j in ['06','07','08']]:
            WrongDeliverisInTrim.append(SupplierPerformance['Amount_WD_6M'][AuditHistory['SupplierId'][i]] - SupplierPerformance['Amount_WD_3M'][AuditHistory['SupplierId'][i]])
            BacklogsInTrim.append(SupplierPerformance['Amount_Backlogs_6M'][AuditHistory['SupplierId'][i]] - SupplierPerformance['Amount_Backlogs_3M'][AuditHistory['SupplierId'][i]])
        else:
            WrongDeliverisInTrim.append((SupplierPerformance['Amount_WD_12M'][AuditHistory['SupplierId'][i]]-SupplierPerformance['Amount_WD_6M'][AuditHistory['SupplierId'][i]])/2)
            BacklogsInTrim.append((SupplierPerformance['Amount_Backlogs_12M'][AuditHistory['SupplierId'][i]]-SupplierPerformance['Amount_Backlogs_6M'][AuditHistory['SupplierId'][i]])/2)

    return WrongDeliverisInTrim,BacklogsInTrim

def generate_chronological_features(AuditHistory,SupplierPerformance):
    LatestQualification,TimeWithoutChange,LastAudit = {t:[] for t in [1,2,3]},list(),list()
    for i in AuditHistory.index:
        last_audit_date,time_since_last_audit,last_qualification = time_until_last_audit(AuditHistory['SupplierId'][i],AuditHistory['DerivativeName'][i],AuditHistory['RecentMonth'][i])
        for t in [1,2,3]:
            if not pd.isna(last_audit_date):
                # print(LatestQualification[t])
                LatestQualification[t].append(last_qualification[t])
                LastAudit.append(time_since_last_audit) 
            else:
                LatestQualification[t].append(pd.NA)
                LastAudit.append(pd.NA) 
    return LatestQualification,LastAudit

def time_until_last_audit(supplier, derivative, recent_month):
    # Convert 'RecentMonth' to datetime format
    recent_month = pd.to_datetime(recent_month)

    # Filter AuditHistory for the given supplier and derivative
    supplier_filter = (AuditHistory['SupplierId'] == supplier) & (AuditHistory['DerivativeName'] == derivative)
    supplier_data = AuditHistory[supplier_filter]

    # Filter data for observations before the given 'RecentMonth'
    recent_month_filter = pd.to_datetime(supplier_data['RecentMonth']) < recent_month
    past_data = supplier_data[recent_month_filter]

    # If there is past data, return the time until the last audit
    if not past_data.empty:
        last_audit_date = pd.to_datetime('2022-01')
        time_since_last_audit = 0
        last_qualification = ''
        for i in past_data.index:
            if pd.to_datetime(supplier_data['RecentMonth'][i]) > last_audit_date:
                last_audit_date = pd.to_datetime(supplier_data['RecentMonth'][i])
                time_since_last_audit = recent_month - last_audit_date
                last_qualification = {T:supplier_data[f'T{T}Color'][i] for T in [1,2,3]}
                
        return last_audit_date,time_since_last_audit,last_qualification

    # If no past data, return a default value (you can adjust this based on your needs)
    return pd.NaT,pd.NaT,pd.NaT # pd.NaT represents a missing or undefined datetime value

def clean_observations(x,y):
    rows_with_missing_x = x[x.isnull().any(axis=1)]

    # Drop rows with missing values from 'x'
    x_cleaned = x.dropna()

    # Drop corresponding rows from 'y' based on 'rows_with_missing_x' index
    y_cleaned = y.loc[x_cleaned.index]
    return x_cleaned,y_cleaned

def evaluate_models(X,y):
    # List of models to evaluate
    models = [
        ('RandomForest', RandomForestClassifier()),
        ('LogisticRegression', LogisticRegression()),
        ('SVM', SVC())
    ]

    # Evaluate each model using cross-validation
    for model_name, model in models:
        scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
        print(f'{model_name}: Mean Accuracy = {round(scores.mean(),5)}, Standard Deviation = {round(scores.std(),5)}')
    
def tuneRandomForest(x,y):
    # Define the parameter grid for Random Forest
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Create a Random Forest Classifier
    rf_classifier = RandomForestClassifier(random_state=42)

    # Define the scoring metric (use appropriate scoring metric for your task)
    scorer = make_scorer(accuracy_score)

    # Perform Grid Search to find the best parameters
    grid_search_rf = GridSearchCV(rf_classifier, param_grid_rf, scoring=scorer, cv=5)
    grid_search_rf.fit(x, y)  # Assuming 'T1' is the target variable for T1 stage

    # Print the best parameters
    print("Best Parameters for Random Forest:", grid_search_rf.best_params_)
    print("Achieved Accuracy on Validation Set:", grid_search_rf.best_score_)

    model = RandomForestClassifier(**grid_search_rf.best_params_)
    model.fit(x,y)
    return model

def tuneLogisticRegression(x,y):
    # Define the parameter grid for Logistic Regression
    param_grid_lr = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2']
    }

    # Create a Logistic Regression Classifier
    lr_classifier = LogisticRegression(max_iter=1000, random_state=42)
    # Perform Grid Search to find the best parameters
    scorer = make_scorer(accuracy_score)

    # Perform Grid Search to find the best parameters
    grid_search_lr = GridSearchCV(lr_classifier, param_grid_lr, scoring=scorer, cv=5)
    grid_search_lr.fit(x,y)

    # Define the parameter grid for Logistic Regression
    param_grid_lr = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2']
    }

    # Create a Logistic Regression Classifier
    lr_classifier = LogisticRegression(max_iter=1000, random_state=42)

    
    grid_search_lr = GridSearchCV(lr_classifier, param_grid_lr, scoring=scorer, cv=5)
    grid_search_lr.fit(x,y)

    print("Best Parameters for Logistic Regression:", grid_search_lr.best_params_)
    print("Achieved Accuracy on Validation Set:", grid_search_lr.best_score_)

    print(grid_search_lr.best_params_)
    model = LogisticRegression(penalty=grid_search_lr.best_params_['penalty'],C=grid_search_lr.best_params_['C'])
    model.fit(x,y)
    return model 

def PredictResult(q1,q2,q3):
    if q3 != 0:
        return q3
    elif q2 != 0:
        return q2
    else:
        return q1