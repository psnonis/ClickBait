clear
/usr/local/spark/bin/spark-shell \
--verbose \
--driver-memory 220G \
--master local[*] \
-i $1
