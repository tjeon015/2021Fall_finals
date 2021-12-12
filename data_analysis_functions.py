import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Function to initialize World bank datasets
def initialize_df(path: str, index: str, cols: list) -> pd.DataFrame:
    """
    Reads the dataset from the given path and then converts it to a dataframe, where it then drops the unwanted columns
    and sets an index based on the user input. It will then transpose the data and return the converted dataframe.

    param path: The path to the dataset csv
    param index: The column from the dataframe which is to be set as the index.
    param cols: The list of column names from the dataframe which are to be dropped.
    return: The transposed dataframe with the given index.

    >>> initialize_df('Data/Maternal Mortality.csv','Years',['Country Code','Indicator Name','Indicator Code'])
    Years  Aruba  Africa Eastern and Southern  ...  Zambia  Zimbabwe
    1960     NaN                          NaN  ...     NaN       NaN
    1961     NaN                          NaN  ...     NaN       NaN
    1962     NaN                          NaN  ...     NaN       NaN
    1963     NaN                          NaN  ...     NaN       NaN
    1964     NaN                          NaN  ...     NaN       NaN
    ...      ...                          ...  ...     ...       ...
    2016     NaN                        408.0  ...   222.0     468.0
    2017     NaN                        398.0  ...   213.0     458.0
    2018     NaN                          NaN  ...     NaN       NaN
    2019     NaN                          NaN  ...     NaN       NaN
    2020     NaN                          NaN  ...     NaN       NaN
    <BLANKLINE>
    [61 rows x 266 columns]

    >>> initialize_df('Maternal Mortality.csv','Years',['Country Code','Indicator Name','Indicator Code']) # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    Traceback (most recent call last):
     ...
    FileNotFoundError: [Errno 2] No such file or directory: 'Maternal Mortality.csv'

    """
    df = pd.read_csv(path)
    df = df.rename(columns={'Country Name': 'Years'})
    df = df.drop(columns=cols)
    df = df.set_index(index)
    df = df.transpose()
    return df


# creates new columns containing income_level based values
def income_assign(df: pd.DataFrame) -> pd.DataFrame:
    """Creates additional columns that show the mean fertility rate for the four country classes: lower, lower-middle,
    upper-middle, high. Also creates an additional 'world' column to represent all classes collectively.

    param df: The flipped fertility rate dataframe
    return: The original dataframe with additional columns that show the mean fertility rate for the four country classes

    >>> mm_flip = initialize_df('Data/Maternal Mortality.csv','Years',['Country Code','Indicator Name','Indicator Code'])
    >>> income_assign(mm_flip)
    Years  Aruba  Africa Eastern and Southern  ...  HIGH   WORLD
    1960     NaN                          NaN  ...   NaN     NaN
    1961     NaN                          NaN  ...   NaN     NaN
    1962     NaN                          NaN  ...   NaN     NaN
    1963     NaN                          NaN  ...   NaN     NaN
    1964     NaN                          NaN  ...   NaN     NaN
    ...      ...                          ...  ...   ...     ...
    2016     NaN                        408.0  ...  11.9  251.15
    2017     NaN                        398.0  ...  11.1  245.65
    2018     NaN                          NaN  ...   NaN     NaN
    2019     NaN                          NaN  ...   NaN     NaN
    2020     NaN                          NaN  ...   NaN     NaN
    <BLANKLINE>
    [61 rows x 271 columns]
    """
    lower_income = ['Afghanistan', 'Liberia', 'Chad', 'Madagascar', 'Mali', 'Uganda', 'Sudan', 'Niger', 'Sierra Leone',
                    'Burundi']
    lower_middle = ['Bangladesh', 'Cameroon', 'Ukraine', 'Zimbabwe', 'Haiti', 'Philippines', 'Tanzania', 'Nicaragua',
                    'Kenya', 'Honduras']
    upper_middle = ['Dominican Republic', 'Ecuador', 'Fiji', 'Thailand', 'Turkey', 'Paraguay', 'Namibia', 'Cuba',
                    'South Africa', 'Belarus']
    high_income = ['Japan', 'Spain', 'Kuwait', 'Sweden', 'Chile', 'Latvia', 'Uruguay', 'Canada', 'United States',
                   'Singapore']
    combined = lower_income + lower_middle + upper_middle + high_income
    df = df.assign(LOWER=df[lower_income].mean(1))
    df = df.assign(LOWER_MIDDLE=df[lower_middle].mean(1))
    df = df.assign(UPPER_MIDDLE=df[upper_middle].mean(1))
    df = df.assign(HIGH=df[high_income].mean(1))
    df = df.assign(WORLD=df[combined].mean(1))
    return df


def correlation_calc(df1, df2, name1, name2):
    """Calculates the correlation coefficients between two dataframes for each country economic class. Then prints out the
    coefficients for each of the four classes.

    :param
        df1 (df): Flipped dataframe of the first dataset being compared
        df2 (df): Flipped dataframe of the second dataset being compared
        name1 (str): Name of the first variable
        name2 (str): Name of the second variable
    :return: Prints out four lines stating the correlation coefficients between the two dataframes at each country class.

    >>> fr_flip = initialize_df('Data/Fertility_Rate.csv','Years',['Country Code','Indicator Name','Indicator Code'])
    >>> fr_flip = income_assign(fr_flip)
    >>> gdp_flip = initialize_df('Data/gdp.csv','Years',['Country Code','Indicator Name','Indicator Code'])
    >>> gdp_flip = income_assign(gdp_flip)
    >>> correlation_calc(fr_flip, gdp_flip, 'Fertility Rate', 'GDP')
    Correlation between Fertility Rate and GDP in LOWER Income Countries:  -0.946
    Correlation between Fertility Rate and GDP in LOWER_MIDDLE Income Countries:  -0.865
    Correlation between Fertility Rate and GDP in UPPER_MIDDLE Income Countries:  -0.794
    Correlation between Fertility Rate and GDP in HIGH Income Countries:  -0.859

    >>> fr_flip = initialize_df('Data/Fertility_Rate.csv','Years',['Country Code','Indicator Name','Indicator Code'])
    >>> gdp_flip = initialize_df('Data/gdp.csv','Years',['Country Code','Indicator Name','Indicator Code'])
    >>> correlation_calc(fr_flip, na, 'Fertility Rate', 'NA') # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    Traceback (most recent call last):
     ...
    NameError: name 'na' is not defined
    """
    category = ['LOWER', 'LOWER_MIDDLE', 'UPPER_MIDDLE', 'HIGH']
    for cat in category:
        corr = df1[cat].corr(df2[cat])
        print('Correlation between {} and {} in {} Income Countries: '.format(name1, name2, cat), corr.round(3))

def rate_plot(df: pd.DataFrame, rate_kind: str) -> None:
    """
    :param df: Dataframe received as an output from initialize_df()
    :param rate_kind: The type of Rate
    :return: None, returns a plot

    >>> mm_flip = initialize_df('Data/Maternal Mortality.csv','Years',['Country Code','Indicator Name','Indicator Code'])
    >>> rate_plot(mm_flip,Maternal_Mortality)
    Traceback (most recent call last):
     ...
    NameError: name 'Maternal_Mortality' is not defined
     """
    plt.figure(figsize=(15, 5), dpi=80)
    plt.plot(df['LOWER'], label = rate_kind + " Rate of Low Income Countries")
    plt.plot(df['HIGH'], label = rate_kind + " Rate of High Income Countries")
    plt.plot(df['WORLD'], label = rate_kind + " Rate World Average")
    plt.xlabel('Years')
    plt.ylabel(rate_kind +' Rate')
    plt.xticks(rotation = 90)
    plt.legend()
    plt.show()

def importing_and_data_cleaning(file_path: str, column_list: list) -> pd.DataFrame:
    """
    Takes the filepath of the directory where the data file is stored. It imports the file in
    '.csv' format and cleans the data to convert it into the required format.
    param file_path: The path to store the data file
    column_list: List of all the columns to be updated in the dataframe
    return: Dataframe updated with the necessary data cleaning processes
    link to avoid warning in the doctest - https://stackoverflow.com/questions/66603854/futurewarning-the-default-value-of-regex-will-change-from-true-to-false-in-a-fu
    >>> importing_and_data_cleaning("Doctest_data/importing_and_cleaning_data.csv",['Total', 'Housing', 'Food', 'Transportation','Clothing', 'Healthcare', 'Child care and Education', 'Miscellaneous'])
      Age of child   Total  Housing  ...  Miscellaneous  Income Category  Year
    0       0 to 2  5490.0   2100.0  ...          540.0       Low Income  1995
    1       3 to 5  5610.0   2080.0  ...          550.0       Low Income  1995
    2       6 to 8  5740.0   2010.0  ...          580.0       Low Income  1995
    3      9 to 11  5770.0   1810.0  ...          610.0       Low Income  1995
    4     12 to 14  6560.0   2020.0  ...          770.0       Low Income  1995
    5     15 to 17  6460.0   1630.0  ...          560.0       Low Income  1995
    <BLANKLINE>
    [6 rows x 11 columns]

    >>> column_list = ['Total', 'Housing', 'Food', 'Transportation','Clothing', 'Healthcare', 'Child care and Education', 'Miscellaneous', 'Unknown']
    >>> child_expenses = importing_and_data_cleaning("Data/US_child_expenditure_data.csv", column_list)
    Traceback (most recent call last):
        ...
    KeyError: 'Unknown'
    """
    df = pd.read_csv(file_path)
    for col_ele in column_list:
        df[col_ele] = df[col_ele].str.replace(",", "", regex=True)
        df[col_ele] = df[col_ele].str.replace("$", "", regex=True)
        df[col_ele] = df[col_ele].astype(float)
    return df


def plotting_child_expense(df: pd.DataFrame, selected_columns: list, categorical_var: list) -> None:
    """
    Function used to plot the cost of different components of child expenditure for the three listed income categories namely,
    low income, medium income and high income.
    df: dataframe storing the child expenditure data
    selected_columns: list columns representing the components of child expenditure to be plotted
    categorical_var: list of income categories used in the study
    return: None

    >>> child_expenses = pd.DataFrame()
    >>> column_list = ['Total', 'Housing', 'Food', 'Transportation','Clothing', 'Healthcare', 'Child care and Education', 'Miscellaneous']
    >>> income_cat = 'Low Income, Medium Income, High Income'
    >>> plotting_child_expense(child_expenses, column_list, income_cat)
    Traceback (most recent call last):
     ...
    KeyError: "None of ['Year', 'Income Category'] are in the columns"


    """
    df.set_index(['Year', 'Income Category'])

    for ele in categorical_var:
        plt.figure(figsize=(10, 6), dpi=80)
        temp = df[df['Income Category'] == ele]
        temp.set_index(['Year'], inplace=True)
        temp = temp.groupby(by='Year').sum()
        plt.plot(temp['Total'], label="Total")
        plt.stackplot(df['Year'].unique(), temp['Housing'], temp['Food'], temp['Transportation'], temp['Clothing'],
                      temp['Healthcare'], temp['Child care and Education'], temp['Miscellaneous'],
                      labels=selected_columns)
        plt.xticks(df['Year'].unique(), rotation=270)
        plt.title("Child Expenditure for " + ele)
        plt.xlabel("Year")
        plt.ylabel("Expenditure on Child")
        plt.legend(bbox_to_anchor=(1.0, 1.0))
        plt.show()

