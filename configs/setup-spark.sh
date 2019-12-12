rm -rf /usr/local/spark/conf/spark-defaults.* /usr/local/spark/conf/log4j.*
ln -sf $(realpath log4j.properties)    /usr/local/spark/conf/log4j.properties
ln -sf $(realpath spark-defaults.conf) /usr/local/spark/conf/spark-defaults.conf
ls -lh /usr/local/spark/conf