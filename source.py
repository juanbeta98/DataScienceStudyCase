'''
Source code for BMW thinktank interview

Juan Betancourt
juan.beta98@gmail.com
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def generate_chronological_features(AuditHistory,SupplierPerformance,T):
    LatestQualification,TimeWithoutChange,LastAudit = list(),list(),list()
    for i in AuditHistory.index:
        last_audit_date,time_since_last_audit,last_qualification = time_until_last_audit(AuditHistory['SupplierId'][i],AuditHistory['DerivativeName'][i],AuditHistory['RecentMonth'][i],T)
        if last_audit_date != pd.NaT:
            LatestQualification.append(last_qualification)
            LastAudit.append(time_since_last_audit) 
    
    return LatestQualification,LastAudit

def time_until_last_audit(supplier, derivative, recent_month,T):
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
                last_qualification = supplier_data[f'T{T}Color'][i]
                
        return last_audit_date,time_since_last_audit,last_qualification

    # If no past data, return a default value (you can adjust this based on your needs)
    return pd.NaT,pd.NaT,pd.NaT # pd.NaT represents a missing or undefined datetime value

