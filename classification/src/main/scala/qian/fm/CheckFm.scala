package qian.fm



import breeze.linalg.DenseVector

import scala.io.Source

object CheckFm {

  val testFile = "C:\\NewCloudMusicProject\\machinelearning\\classification\\src\\main\\resources\\x"
  var testData: Array[(Int, Array[Double])] = _

  def checkGrad(): Unit = {
    val trainFm = new TrainFm(2)
    trainFm.setData(testData)
    trainFm.setDim()
    trainFm.initFmModel()

    val fmModel = trainFm.fmModel

    val testY = testData.apply(3)._1
    val testX = new DenseVector(testData(3)._2)
    println(testY)
    println(testX)

    println("检查z")
    println(fmModel.getZ(testX))   //正确

    println("检查predict")
    println(fmModel.predictScore(testX))   //正确  0.9623121094913941

    println("检查loss")
    println(fmModel.getLoss(testX, testY))   //正确  ln  3.27


    val r = fmModel.getGrad(testX, testY)
    val DB = r._1
    val DW = r._2
    val DV = r._3

    println("检查db\n")
    println(DB)


    println("检查dw\n")
    println(DW)
    fmModel.checkGradW(testX, testY, 0.00001, 4)
    fmModel.checkGradW(testX, testY, 0.1, 5)
    fmModel.checkGradW(testX, testY, 0.01, 6)


    println("检查dv")
    println(DV)
    fmModel.checkGradV(testX, testY, 0.0001, 4, 0)
    fmModel.checkGradV(testX, testY, 0.00001, 4, 1)
    fmModel.checkGradV(testX, testY, 0.001, 5, 0)
  }

  def testTrain(): Unit ={
    val trainFm = new TrainFm(4)
    trainFm.setData(testData)
    trainFm.setDim()
    trainFm.initFmModel()

    for (i <- 0.until(100)) {
      trainFm.stochasticTrain(1, 0.001)
      println(trainFm.getLoss())
    }


    for (d <- this.testData) {
      println(d._1)
      println(trainFm.fmModel.predictScore(DenseVector(d._2)))
    }

    println("训练集正确率:" + trainFm.testAccuracy())

  }


  def main(args: Array[String]): Unit = {
    readFile()
    //checkGrad()
    testTrain()
  }


  def readFile(): Unit = {
    this.testData = Source.fromFile(testFile).getLines().toArray.map{ l =>
      val lines = l.split("\t")
      val y = lines(0).toInt
      val x = lines(1).toCharArray.map(x => x.toString.toDouble)
      (y, x)
    }.filter(x => x._1 <= 1)
//    this.testData.foreach(x =>
//      println(x._1, x._2.mkString(","))
//    )
    println("加载数据完成")
  }
}
