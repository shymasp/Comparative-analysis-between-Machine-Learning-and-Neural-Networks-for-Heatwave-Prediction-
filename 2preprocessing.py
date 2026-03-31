import pandas as pd

# Paths to Chennai datasets
file_1 = "C:/Users/SHYMA SUNDARAM/Desktop/heatwave prediction system for chennai/data/raw/chennai_2010_2011.csv"
file_2 = "C:/Users/SHYMA SUNDARAM/Desktop/heatwave prediction system for chennai/data/raw/chennai_2012_2013.csv"
file_3 = "C:/Users/SHYMA SUNDARAM/Desktop/heatwave prediction system for chennai/data/raw/chennai_2014_2015.csv"
file_4 = "C:/Users/SHYMA SUNDARAM/Desktop/heatwave prediction system for chennai/data/raw/chennai_2016_2017.csv"
file_5 = "C:/Users/SHYMA SUNDARAM/Desktop/heatwave prediction system for chennai/data/raw/chennai_2018_2019.csv"
file_6 = "C:/Users/SHYMA SUNDARAM/Desktop/heatwave prediction system for chennai/data/raw/chennai_2020_2021.csv"
file_7 = "C:/Users/SHYMA SUNDARAM/Desktop/heatwave prediction system for chennai/data/raw/chennai_2022_2023.csv"

# Read datasets
df1 = pd.read_csv(file_1)
df2 = pd.read_csv(file_2)
df3 = pd.read_csv(file_3)
df4 = pd.read_csv(file_4)
df5 = pd.read_csv(file_5)
df6 = pd.read_csv(file_6)
df7 = pd.read_csv(file_7)

# Combine datasets
combined_chennai = pd.concat([df1, df2,df3,df4,df5,df6,df7], ignore_index=True)

# Save combined dataset
combined_chennai.to_csv(
    "C:/Users/SHYMA SUNDARAM/Desktop/heatwave prediction system for chennai/data/clean/chennai_combined_2010_2023.csv",index=False)

# Outputs
print("Chennai datasets merged successfully!")
print("Shape:", combined_chennai.shape)
print(combined_chennai.head())
print(combined_chennai.isnull().sum())