import pytest
import os
import csv
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark_session():
    spark = SparkSession.builder.appName("test").getOrCreate()
    yield spark
    spark.stop()


# @pytest.fixture(scope="session")
# def create_csv_files(tmp_path):
#     # Define CSV files and their content
#     files_content = {
#         "test_data1.csv": [("file_name", "column_with_null"), ("row1_col1", "row1_col2")],
#         "test_data2.csv": [("headerA", "headerB"), ("rowA_colA", "rowA_colB")],
#     }

#     # Create CSV files
#     for file_name, content in files_content.items():
#         file_path = os.path.join(tmp_path, file_name)
#         with open(file_path, "w", newline="") as file:
#             writer = csv.writer(file)
#             writer.writerows(content)
