from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import geopandas as gpd
import pandas as pd
import os
os.environ["PROJ_LIB"] = "/opt/conda/miniconda3/share/proj"


def load_raw_data(spark, path):
    df = spark.read.parquet(path)
    df = df.withColumn("tpep_pickup_datetime", F.col("tpep_pickup_datetime").cast("timestamp")) \
           .withColumn("tpep_dropoff_datetime", F.col("tpep_dropoff_datetime").cast("timestamp"))
    df = df.dropDuplicates()
    df = df.withColumn("trip_id", F.monotonically_increasing_id())
    return df

def create_location_dim(spark, zones_path, zone_lookup_path):
    zones_gdf = gpd.read_file(zones_path)
    zone_lookup_pd = pd.read_csv(zone_lookup_path)
    location_pd = zones_gdf.merge(zone_lookup_pd, on='LocationID') \
                           .drop(columns=['Borough', 'Zone']) \
                           .drop_duplicates(subset=['LocationID']) \
                           .reset_index(drop=True)
    location_pd["geometry"] = location_pd["geometry"].apply(lambda geom: geom.wkt if geom else None)
    return spark.createDataFrame(location_pd)

def create_datetime_dim(spark, df, datetime_col, prefix):
    dt_dim = df.select(datetime_col).dropDuplicates()
    dt_dim = dt_dim.withColumn(f"{prefix}_hour", F.hour(datetime_col)) \
                   .withColumn(f"{prefix}_day", F.dayofmonth(datetime_col)) \
                   .withColumn(f"{prefix}_month", F.month(datetime_col)) \
                   .withColumn(f"{prefix}_year", F.year(datetime_col)) \
                   .withColumn(f"{prefix}_weekday", F.dayofweek(datetime_col) - F.lit(1))

    dt_dim = dt_dim.withColumn(f"{prefix}_datetime_id", F.monotonically_increasing_id())
    return dt_dim

def create_rate_code_dim(spark):
    rate_code_data = [
        (1, "Standard rate"),
        (2, "JFK"),
        (3, "Newark"),
        (4, "Nassau or Westchester"),
        (5, "Negotiated fare"),
        (6, "Group ride")
    ]
    return spark.createDataFrame(rate_code_data, schema=["rate_code_id", "rate_code_name"])

def create_payment_type_dim(spark):
    payment_type_data = [
        (1, "credit card"),
        (2, "cash"),
        (3, "no charge"),
        (4, "dispute"),
        (5, "unknown"),
        (6, "voided trip")
    ]
    return spark.createDataFrame(payment_type_data, schema=["payment_type_id", "payment_type_name"])

def create_fact_table(df, rate_code_dim, pickup_dt_dim, dropoff_dt_dim, payment_type_dim, location_dim):
    # Join with rate_code_dim on RatecodeID
    fact = df.join(rate_code_dim, df["RatecodeID"] == rate_code_dim["rate_code_id"], "left")
    # Join with pickup datetime dimension
    fact = fact.join(pickup_dt_dim, on="tpep_pickup_datetime", how="left")
    # Join with dropoff datetime dimension
    fact = fact.join(dropoff_dt_dim, on="tpep_dropoff_datetime", how="left")
    # Join with payment type dimension on payment_type
    fact = fact.join(payment_type_dim, fact["payment_type"] == payment_type_dim["payment_type_id"], "left")
    
    # For location dimension, join twice with aliasing
    loc_pickup = location_dim.withColumnRenamed("LocationID", "PULocationID_dim")
    fact = fact.join(loc_pickup, fact["PULocationID"] == F.col("PULocationID_dim"), "left")
    
    loc_dropoff = location_dim.withColumnRenamed("LocationID", "DOLocationID_dim")
    fact = fact.join(loc_dropoff, fact["DOLocationID"] == F.col("DOLocationID_dim"), "left")
    
    # Select desired columns for the final fact table.
    fact = fact.select(
        "trip_id", "VendorID", "pickup_datetime_id", "dropoff_datetime_id",
        "passenger_count", "trip_distance", "rate_code_id", "store_and_fwd_flag",
        "PULocationID", "DOLocationID", "payment_type_id", "fare_amount", "extra",
        "mta_tax", "tip_amount", "tolls_amount", "total_amount", "congestion_surcharge",
        "Airport_fee"
    )
    return fact

# def write_table(df, hdfs_path):
#     df.write.mode("overwrite").parquet(hdfs_path)

def main():
    spark = SparkSession.builder.appName("taxi_etl").getOrCreate()
    
    # Define file paths (adjust as needed)
    raw_data_path = "data/yellow_tripdata_2024-01.parquet"
    zones_path = "data/taxi_zones/taxi_zones.shp"
    zone_lookup_path = "data/taxi_zone_lookup.csv"
    
    # Load and transform data
    df = load_raw_data(spark, raw_data_path)
    location_dim = create_location_dim(spark, zones_path, zone_lookup_path)
    pickup_dt_dim = create_datetime_dim(spark, df, "tpep_pickup_datetime", "pickup")
    dropoff_dt_dim = create_datetime_dim(spark, df, "tpep_dropoff_datetime", "dropoff")
    rate_code_dim = create_rate_code_dim(spark)
    payment_type_dim = create_payment_type_dim(spark)
    fact_table = create_fact_table(df, rate_code_dim, pickup_dt_dim, dropoff_dt_dim, payment_type_dim, location_dim)
    
    # Write out dimension and fact tables to different project and bucket
    fact_table.write.mode("overwrite").parquet("gs://uber-data-engineer-project-iris/taxi_etl/fact_table")
    location_dim.write.mode("overwrite").parquet("gs://uber-data-engineer-project-iris/taxi_etl/location_dim")
    pickup_dt_dim.write.mode("overwrite").parquet("gs://uber-data-engineer-project-iris/taxi_etl/pickup_dt_dim")
    dropoff_dt_dim.write.mode("overwrite").parquet("gs://uber-data-engineer-project-iris/taxi_etl/dropoff_dt_dim")
    rate_code_dim.write.mode("overwrite").parquet("gs://uber-data-engineer-project-iris/taxi_etl/rate_code_dim")
    payment_type_dim.write.mode("overwrite").parquet("gs://uber-data-engineer-project-iris/taxi_etl/payment_type_dim")
    spark.stop()


if __name__ == "__main__":
    main()
