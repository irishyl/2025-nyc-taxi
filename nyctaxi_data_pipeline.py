import pandas as pd

df = pd.read_csv("data/uber_data.csv")

# inspecting date type and modify to suitable data type
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
# print(df.info())

df = df.drop_duplicates().reset_index(drop=True)
df['trip_id'] = df.index

# creating dim tables (structure in data model.jpeg)
# datetime_dim
datetime_dim = df[['tpep_pickup_datetime', 'tpep_dropoff_datetime']].drop_duplicates().reset_index(drop=True)