clear
/usr/local/spark/bin/spark-submit \
--verbose \
--packages org.apache.spark:spark-avro_2.11:2.4.3 \
--driver-memory 240G \
--master local[32] \
lr.py