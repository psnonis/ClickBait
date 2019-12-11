clear
/usr/local/spark/bin/spark-submit \
--verbose \
--driver-memory 220G \
--master local[*] \
$1
