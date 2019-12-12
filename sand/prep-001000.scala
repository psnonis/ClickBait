class Common {
    
    import org.apache.spark.sql.{SparkSession, DataFrame}
    
    var prefix  : String       = "data"
    var spark   : SparkSession = null
    
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
}

object Interaction extends Common {
  
    import org.apache.spark.ml.{Pipeline, PipelineModel}
    import org.apache.spark.ml.feature.{Interaction, VectorAssembler}
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
    
    def allJoinInteract(subset: String, iStep: String) {

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

Interaction.setupSpark(application = "interaction")

var min = 1000

Interaction.allJoinInteract("train", f"normed.masked-${min}%06d.encode")
Interaction.allJoinInteract("tests", f"normed.masked-${min}%06d.encode")
Interaction.allJoinInteract("valid", f"normed.masked-${min}%06d.encode")

Interaction.allPackFeatures("train", f"normed.masked-${min}%06d.encode.action")
Interaction.allPackFeatures("tests", f"normed.masked-${min}%06d.encode.action")
Interaction.allPackFeatures("valid", f"normed.masked-${min}%06d.encode.action")