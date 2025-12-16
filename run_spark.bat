@echo off
spark-submit ^
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.6 ^
  consumer\spark_job.py

pause