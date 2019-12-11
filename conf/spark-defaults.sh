rm -rf /usr/local/spark/conf/spark-defaults.*
ln -sf $(realpath spark-defaults.conf) /usr/local/spark/conf/spark-defaults.conf
ls -lh /usr/local/spark/conf