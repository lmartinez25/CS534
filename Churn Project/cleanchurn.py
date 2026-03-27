import pandas as pd
import matplotlib.pyplot as plt

#Clean and Inspect Data

# Step 1: Load and inspect original dataset
# Dataset starts with 7042 rows, 21 columns.
# Columns include: 3 numerical features (tenure, MonthlyCharges, TotalCharges),
# 17 categorical and 1 unnecessary (customer id) that we will drop
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
pd.set_option('display.max_columns', None)  # ensures all columns show when printing out data
df.head()

def inspect():
    print("\nData Size: ", df.shape, "\n")
    print("Column Names: ", "\n", df.columns, "\n")


    # Step 2: Look for null values and incorrect type assignment
    # We find there are no null values. However, TotalCharges which is numerical is shown as string type.
    # We encountered error when converting to float due to entries that were ' '. There were 11 such
    # entries that were connected with 11 entries in tenure column that were 0. We replaced the ' '
    # entries with 0 since they mean that customers have not yet paid since they just signed up.
    df.info()
    df[df['TotalCharges'] == ' ']
    len(df[df['TotalCharges'] == ' '])
    len(df[df['tenure'] == 0])


# Step 3: Edit data according to our findings
# We drop customerID, replace TotalCharges == ' ' with 0, and convert TotalCharges to float
df = df.drop(columns = 'customerID')
df['TotalCharges'] = df['TotalCharges'].replace(' ', '0.0')
df['TotalCharges'] = df['TotalCharges'].astype(float)

def main():
    # Step 4: Inspect categorical data and plot
    # Churn is unbalanced (73% No), Categories have mostly 2 or 3 categories (one has 4)
    categories = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                  'PhoneService', 'MultipleLines', 'InternetService',
                  'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                  'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                  'PaymentMethod','Churn']
    for i in categories:
        print(df[i].value_counts(normalize=True) * 100, '\n')

    # Step 5: Inspect numerical data
    # check distributions (they are all skewed, not normal or uniform).
    # do we need to normalize?
    print("Statistics of Numerical data: \n", df.describe())
    df['tenure'].plot(kind='hist')
    plt.title('Histogram of Tenure')
    plt.show()
    df['MonthlyCharges'].plot(kind='hist')
    plt.title('Histogram of Monthly Charges')
    plt.show()
    df['TotalCharges'].plot(kind='hist')
    plt.title('Histogram of Total Charges')
    plt.show()


if __name__ == "__main__":
    inspect()
    main()

