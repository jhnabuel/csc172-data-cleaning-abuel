## Student Information
- Name: John Christian Niño T. Abuel
- Course Year: BSCS 4
- Date: 2025-09-29

## Dataset 
- Source: https://archive.ics.uci.edu/dataset/2/adult
- Name: Adult Dataset

## Issues found
- Missing values: Under features workclass, occupation, and native-country
- Duplicates: 23 duplicate rows
- Inconsistencies: Whitespace before the data value and cases are not uniform (Married vs married) among features

## Cleaning steps
1. Missing values: Dropped rows with missing values for workclass, occupation, and native-country
2. Duplicates: Dropped duplicates
3. Inconsistencies: Removed whitespace and uniformed cases among features
4. Outliers: Optional (kind of), but for convenience, removed fnlwgt and log-transform skewed numeric columns such as capital_gain and capital_loss

## AI prompts used
- Prompt 1: "from the adult dataset from UCI, it seems to me that this file doesn't have column headers. I am importing this into a python notebook through pd.read_csv, do I use df.columns = ['ColA', 'ColB'] so that each feature is properly defined?"
- Generated response: 
column_names = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income"
]
df = pd.read_csv("/mnt/data/adult.data", header=None, names=column_names)
Here’s what’s happening:
header=None tells pandas there are no headers in the file.
names=column_names assigns the proper feature names.
That way you don’t need to manually do df.columns = [...] after loading.

- Prompt 2: Through pandas, how do I drop the rows with empty values from these columns?
- Generated response: 
    Replace " ?" with NaN in the specific columns
    df[['workclass', 'occupation', 'native_country']] = df[['workclass', 'occupation', 'native_country']].replace(' ?', pd.NA)

    Drop rows where any of those columns are NaN
    df = df.dropna(subset=['workclass', 'occupation', 'native_country'])


- Prompt 3: How do I count the number of rows in the entire dataframe before and after dropping rows with empty features? And also, how do I count duplicate rows in the dataframe and then drop them
- Generated response:
    1. Count number of rows before and after dropping missing rows
        Total rows before dropping
    print("Rows before:", len(df))

    Drop rows with missing values in specific columns
    df_cleaned = df.dropna(subset=['workclass', 'occupation', 'native_country'])

    Total rows after dropping
    print("Rows after:", len(df_cleaned))

    2. Count duplicate rows
    duplicate_count = df_cleaned.duplicated().sum()
    print("Number of duplicate rows:", duplicate_count)

    3. Drop duplicate rows
    Drop duplicates
    df_no_duplicates = df_cleaned.drop_duplicates()

    Verify new row count
    print("Rows after dropping duplicates:", len(df_no_duplicates))

- Prompt 4: How do I address the inconsistent formatting in this dataset? Especially whitespace issues or slight differences in formatting?
- Generated response:
    Excellent question — the UCI Adult dataset is notorious for these formatting quirks. If you don’t fix them, you’ll end up with messy categories (like " Private" vs "Private") being treated as different values.

    Here’s how to systematically address inconsistent formatting:

    1. Strip Whitespace from Strings

    Almost all categorical columns contain leading spaces after the commas.

    Strip whitespace from all string columns
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)


    This ensures " Private" → "Private", " United-States" → "United-States", etc.

    2. Standardize Case (optional)

    To avoid "Male" vs "male" being separate, you can force lowercase:

    df = df.applymap(lambda x: x.lower().strip() if isinstance(x, str) else x)


    After this, "Private" → "private", "Self-emp-not-inc" → "self-emp-not-inc".
    This makes comparisons more consistent.

    3. Replace Known Inconsistent Variants

    Sometimes categories have slight variations (depends on dataset version). For example:

    "United-States" vs "United States"

    "Self-emp-not-inc" vs "Self-emp-inc" (valid but can look inconsistent)

    If you find such inconsistencies, standardize them with replace:

    df['native_country'] = df['native_country'].replace({
        'United-States': 'United States'
    })

    4. Verify Unique Values

    After cleaning, check that categories look consistent:

    print(df['workclass'].unique())
    print(df['occupation'].unique())
    print(df['native_country'].unique()[:10])  # first 10 countries

- Prompt 5: What about this: df['Category'] = df['Category'].str.strip().str.lower(), doesn't this strip whitespace and standardize case already?
- Generated response
    Yes — you’ve got it exactly right 
    That line:

    df['Category'] = df['Category'].str.strip().str.lower()

    does two important things in one go:
    .str.strip() → removes leading and trailing whitespace (e.g., " Private" → "Private").
    .str.lower() → standardizes everything to lowercase (e.g., "Private" → "private").

    So after this, " Private", "private", "Private" → all become "private".

    The difference between your example and the applymap approach I showed earlier:
    Your approach: applies to one specific column (Category).
    .applymap() approach: applies to all string columns in the DataFrame at once.

    If you want to fix all categorical features at once, you could do:

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip().str.lower()

    This will handle whitespace + case standardization for every categorical column, without needing to write the same line multiple times.

- Prompt 6: And for the outlier detection and treatment, how should I approach this?
- Generated response: 
    Great question — outlier detection and treatment in the Adult dataset really depends on what you plan to do with the data (exploratory analysis vs. machine learning). Let’s break it down by feature and then talk strategy.
    Where outliers typically show up in the Adult dataset:
    1. age
        Range: 17 → 90+
        Outliers: people at 90+ are extremely rare. Not an error, but can skew models.

    2. fnlwgt (final sampling weight)
        Huge values (tens of thousands to millions).
        Looks like an outlier but is not an error — it’s a weighting factor used by the Census Bureau.
        Many people simply drop this column for predictive modeling.
    3. capital_gain and capital_loss
        Most values are 0, but a few extreme values exist:
        capital_gain has values like 99999 (artificial upper bound).
        capital_loss also has some unusually large values.
        These are genuine but very skewed — behave like outliers.
    4. hours_per_week
        Range: 1 → 99
        Most people work ~40 hours.
        Outliers: people working 1 hour/week or 99 hours/week. These are rare but plausible.

- Prompt 7: Generate pandas code to log-transform capital_gain and capital_loss, dropping fnlwgt
- Generated response:
    import numpy as np

    Drop fnlwgt column
    df = df.drop(columns=['fnlwgt'])

    Log-transform skewed numeric columns
    df['capital_gain'] = np.log1p(df['capital_gain'])
    df['capital_loss'] = np.log1p(df['capital_loss'])

## Results
- Rows before: 32561 
- Rows after: 30162 

Video: