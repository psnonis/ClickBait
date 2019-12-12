class Common {
    
    import org.apache.spark.sql.{SparkSession, DataFrame}

    import reflect.io._, Path._
    import scala.collection.mutable.HashMap

    val frames = new HashMap[String,DataFrame]()

    var prefix  : String       = "data"
    var spark   : SparkSession = null
    
    val num_features: Array[String] = (1 to 13).map(f => f"n$f%02d"         ).toArray
    val std_features: Array[String] = (1 to 13).map(f => f"n$f%02d_standard").toArray
    val cat_features: Array[String] = (1 to 26).map(f => f"c$f%02d"         ).toArray
    
    val cat_uncommon = new HashMap[String,Array[Any]]()
    val cat_frequent = new HashMap[String,Array[Any]]()
    val cat_distinct = new HashMap[String,Array[Any]]()

    var num_measures: DataFrame = null
    var cat_measures: DataFrame = null
    
    def timePrint(message: String) {
        
        var timeStamp = java.time.format.DateTimeFormatter.ofPattern("HH:mm:ss").format(java.time.LocalDateTime.now)
        println(f"${timeStamp} : ${message}")
    }
   
    def setupSpark(application: String = "w261", master: String = "local[*]", memory: String = "220G") {

        import org.apache.spark.{SparkContext,SparkConf}
        
        timePrint("Starting Spark Initialization")

        SparkSession.builder.getOrCreate().stop()

        spark = SparkSession
            .builder()
            .master(master)
            .appName(application)
            .config("spark.driver.memory", memory)
            .getOrCreate()

        timePrint("Stopping Spark Initialization\n")
    }
    
    def importData(location: String = "data", clean: Boolean = false)
    {
        import org.apache.spark.sql.types.{StructType,StructField,IntegerType,FloatType,StringType}

        timePrint("Starting Data Import")

        prefix = f"$location"

        if (clean) {
        }

        var frame = f"whole"
        var train = f"${location}/${frame}.zip"
        var whole = f"${location}/${frame}.parquet"

        if (!File(whole).exists) {

            var schema = StructType(Seq(StructField("label", IntegerType, true)))
            
            num_features.foreach(column => {

                schema = schema.add(StructField(column,  FloatType, true))
            })

            cat_features.foreach(column => {

                schema = schema.add(StructField(column, StringType, true))
            })
            
            val criteo = spark.read.format("csv")
                        .option("header", "false")
                        .option("delimiter", "\t")
                        .schema(schema)
                        .load(train)

            criteo.write.parquet(whole)
        }
        
        frames(frame) = spark.read.parquet(whole)
        
        /*
        location.toDirectory.dirs.map(_.path)
            .filter( name => name matches f"""$location/(whole|train|valid|tests).parquet.*""")
            .toList.sorted.reverse.foreach ((path) => {

            timePrint(f"Loading -> $path")

            var sub       = path.split(f"$location/").last.split(".parquet").head
            var frame     = path.split(f"$location/$sub.parquet").last.stripSuffix(".")

            frames(f"$sub.$frame") = spark.read.parquet(path)

        })
        */
        
        timePrint("Stopping Data Import\n")
    }
    
    def splitsData(ratios: Array[Double] = Array(0.8, 0.1, 0.1)) {

        timePrint("Starting Data Splits")
        
        val splits  = frames("whole").randomSplit(ratios, seed = 2019)
        val indexes = Array("train", "tests", "valid")

        for ((split, subset) <- splits zip indexes) {
            var path = f"$prefix/$subset.parquet"

            if  (!path.toDirectory.exists) {
                timePrint(f"Saving -> $path")
                split.write.parquet(path)
            }
        }
        
        timePrint("Stopping Data Splits\n")
    }
}

object Interaction extends Common {
  
    import org.apache.spark.ml.{Pipeline, PipelineModel}
    import org.apache.spark.ml.feature.{ChiSqSelector, Interaction, VectorAssembler, Imputer, StandardScaler}
    import org.apache.spark.sql.{DataFrame}
    import org.apache.spark.ml.linalg.{SparseVector,DenseVector}
    import org.apache.spark.sql.functions.{col}

    import reflect.io._, Path._
    import sys.process._

    var iFile: String = null
    var oStep: String = null
    var oFile: String = null
    var oPipe: String = null
    
    def taskStarting(step: String, message: String, subset: String, iStep: String): DataFrame = {
    
        iFile = f"$prefix/$subset.parquet.$iStep"
        oStep = f"$iStep.$step"
        oFile = f"$prefix/$subset.parquet.$oStep"
        oPipe = f"$prefix/model.pickled.$oStep"

        timePrint(f"Starting $message : iFile = $iFile")
        
        if (iFile.toDirectory.exists && !f"${iFile}/_SUCCESS".toFile.exists) {
            f"rm -rf ${iFile}" !
        }

        if (oFile.toDirectory.exists && !f"${oFile}/_SUCCESS".toFile.exists) {
            f"rm -rf ${oFile}" !
        }
        
        if (iFile.toDirectory.exists && !oFile.toDirectory.exists) {
            return spark.read.parquet(iFile)
        }
        else {
            return null
        }
    }

    def taskStopping(step: String, message: String, subset: String, oData: DataFrame) {

        if (oData != null && !oFile.toDirectory.exists) {
            oData.write.parquet(oFile)
        }

        timePrint(f"Stopping $message : oFile = $oFile\n")
    }
    
    def numStandardize (subset: String, iStage: String, fit: Boolean = false) {
        
        var iData: DataFrame = taskStarting("scaled", "Numerical Data Standardization", subset, iStage)
        var oData: DataFrame = null
        var xPipe: PipelineModel = null

        if (iData != null) {

            if (fit == true)
            {
                val imputer   = new Imputer().setInputCols(num_features).setOutputCols(std_features).setStrategy("median")
                val assembler = new VectorAssembler().setInputCols(std_features).setOutputCol("imp_features")
                var scaler    = new StandardScaler().setInputCol("imp_features").setOutputCol("std_features")
                var pipeline  = new Pipeline().setStages(Array(imputer,assembler,scaler))

                xPipe = pipeline.fit(iData)
                xPipe.write.overwrite().save(oModel)
            }
            else
            {
                xPipe = PipelineModel.load(oModel)
            }

            oData = xPipe.transform(iData)
        }

        taskStopping("scaled", "Numerical Data Standardization", subset, oData)
    }

    def catMaskUncommon (subset: String, iStage: String, fit: Boolean = false, threshold: Int = 1000) {
        
        var iData: DataFrame = taskStarting(f"masked-$threshold", "Mask Uncommon Categories", subset, iStage)
        var oData: DataFrame = null

        if (iData != null) {

            iData  = iData.na.fill("deadbeef", cat_features).cache()
            
            if (fit == true)
            {
                val frequent = f"count >= $threshold"
                val uncommon = f"count <  $threshold"
                
                var frequent_total = 0
                var uncommon_total = 0

                cat_features.foreach(feature => {

                    var count: DataFrame  = iData.select( feature).groupBy(feature).count()
                    cat_frequent(feature) = count.filter(frequent).sort(col("count").desc).select(feature).collect.map(row => row(0).asInstanceOf[String])
                    cat_uncommon(feature) = count.filter(uncommon).sort(col("count").desc).select(feature).collect.map(row => row(0).asInstanceOf[String])
                //  cat_distinct(feature) = cat_uncommon(feature) + cat_frequent(feature)
                    
                    frequent_total += cat_frequent(feature).size
                    uncommon_total += cat_uncommon(feature).size
                    
                    timePrint("$feature > found ${cat_frequent(feature).size}%7d frequent and ${cat_uncommon(feature).size}%7d uncommon > ${frequent_total}%5d total frequent")
                })
            }
            else
            {
                
            }
        }

        taskStopping(f"masked-$threshold", "Mask Uncommon Categories", subset, oData)
    }    
    
    def allJoinInteract(subset: String, iStep: String) {

        /*
            This had to be written in Scala as PySpark implementation of SparkML interactions transformer was not available.
            Add interactions between the categorical indicators and numerical features.
            Use SparkML Interactions feature transformer to calculate the product of all 13 numerical features with the arbitary number of categorical indicators.
            
            Input  : std_features and cat_features
            Output : cxn_features SparseVector
        */
        
        var iData: DataFrame = taskStarting("action", "Numerical vs Categorical Interactions", subset, iStep)
        var oData: DataFrame = null
        
        if (iData != null) {
            val interaction = new Interaction()
                .setInputCols(Array("std_features", "cat_features"))
                .setOutputCol("cxn_features")

            oData = interaction.transform(iData)
        }

        taskStopping("action", "Numerical vs Categorical Interactions", subset, oData)
    }

    def allPackFeatures(subset: String, iStep: String) {

        var iData: DataFrame = taskStarting("packed", "Final Feature Pack", subset, iStep)
        var oData: DataFrame = null
        var width: Int       = 0
        
        if (iData != null) {
            val assembler = new VectorAssembler()
                .setInputCols(Array("std_features", "cat_features", "cxn_features"))
                .setOutputCol("features")
            
            oData = assembler.transform(iData)

            width = oData.first().getAs[SparseVector]("features").size

            oFile = f"${oFile}-${width}%06d"
            
            timePrint(f"Final Feature Count = ${width}")
        }

        taskStopping("packed", "Final Feature Pack", subset, oData)
    }
}