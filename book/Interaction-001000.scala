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
    import org.apache.spark.sql.functions.{col}

    import reflect.io._, Path._

    var iFrame: String = null
    var oStage: String = null
    var oFrame: String = null
    var oModel: String = null
    
    def taskStarting(stage: String, message: String, subset: String, iStage: String): DataFrame = {
    
        iFrame = f"$prefix/$subset.parquet.$iStage"
        oStage = f"$iStage.$stage"
        oFrame = f"$prefix/$subset.parquet.$oStage"
        oModel = f"$prefix/model.pickled.$oStage"

        timePrint(f"Starting $message : iFrame = $iFrame")
        
        if (iFrame.toDirectory.exists && !oFrame.toDirectory.exists) {
            return spark.read.parquet(iFrame)
        }
        else {
            return null
        }
    }

    def taskStopping(stage: String, message: String, subset: String, oData: DataFrame) {

        if (oData != null) {
            oData.write.parquet(oFrame)
        }

        timePrint(f"Stopping $message : oFrame = $oFrame\n")
    }
    
    def allJoinInteract(subset: String, iStage: String) {

        var iData: DataFrame = taskStarting("action", "Numerical vs Categorical Interactions", subset, iStage)
        var oData: DataFrame = null
        
        if (iData != null) {
            val interaction = new Interaction()
                .setInputCols(Array("std_features", "cat_features"))
                .setOutputCol("cxn_features")

            oData = interaction.transform(iData)
        }

        taskStopping("action", "Numerical vs Categorical Interactions", subset, oData)
    }

    def allPackFeatures(subset: String, iStage: String) {

        var iData: DataFrame = taskStarting("packed", "Final Feature Pack", subset, iStage)
        var oData: DataFrame = null
        
        if (iData != null) {
            val assembler = new VectorAssembler()
                .setInputCols(Array("std_features", "cxn_features"))
                .setOutputCol("features")
            
            oData = assembler.transform(iData)
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