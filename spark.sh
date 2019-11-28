docker run -d -p 8888:8888 -p 4040:4040 -p 4041:4041 \
--name spark \
--user root \
--group-add root \
-d \
-e JUPYTER_ENABLE_LAB=yes \
-e GRANT_SUDO=yes \
-e GEN_CERT=yes \
-v $PWD:/home/jovyan/work \
jupyter/all-spark-notebook \
start-notebook.sh \
--LabApp.password='sha1:39ea6f32055d:7b97c2c76e9b554215a7056e595fda50283df281'
