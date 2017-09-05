package qian.fm

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Created by qianwei on 2017/9/5.
  * 训练FM 模型
  */
class TrainFm(k: Int) extends Serializable{

  var fmModel:FM = null
  var data: Array[(Int, Array[Double])] = _
  var dim:Int = 0

  def setData(data: Array[(Int, Array[Double])]): Unit ={
    this.data = data
  }

  def setDim(): Unit = {
    this.dim = this.data(0)._2.length
  }

  def initFmModel(): Unit = {
    this.fmModel = new FM(this.dim, this.k)
  }

  def train(itNum: Int, stepSize: Double, reg:Double = 0.0): Unit = {
    for (i <- 0.until(itNum)) {
      val r = this.trainStep(reg)
      val sumGradB = r._1
      val sumGradW = r._2
      val sumGradV = r._3
      this.fmModel.updateParam(sumGradB, sumGradW, sumGradV, stepSize)
    }
  }

  def stochasticTrain(itNum: Int, stepSize: Double, reg: Double = 0.0): Unit ={
    for (i <- 0.until(itNum)) {
      val stepSize2 = stepSize * (math.log(itNum - i + 2.0) / math.log(itNum + 2.0))
      println("当前步伐:" + stepSize2)
      for (oneRow <- data) {
        val tempX = new DenseVector[Double](oneRow._2)
        val r = this.fmModel.getGrad(tempX, oneRow._1, reg)
        val sumGradB = r._1
        val sumGradW = r._2
        val sumGradV = r._3
        this.fmModel.updateParam(sumGradB, sumGradW, sumGradV, stepSize2)
      }
      println(this.getLoss())
    }
  }


  def trainStep(reg: Double): (Double, DenseVector[Double], DenseMatrix[Double]) ={
    var grad_b = 0.0
    var grad_w = DenseVector.zeros[Double](this.dim)
    var grad_v = DenseMatrix.zeros[Double](this.dim, this.k)
    for (oneRow <- data) {
      val tempX = new DenseVector[Double](oneRow._2)
      val r = this.fmModel.getGrad(tempX, oneRow._1, reg)
      val b = r._1
      val w = r._2
      val v = r._3
      grad_b = grad_b + b
      grad_w = grad_w + w
      grad_v = grad_v + v
    }
    (grad_b, grad_w, grad_v)
  }

  def getLoss(): Double = {
    var loss = 0.0
    for (d <- this.data) {
      val x = new DenseVector[Double](d._2)
      loss += this.fmModel.getLoss(x, d._1)
      //println(d)
      //println(loss)
    }
    loss
  }

  def testAccuracy(threshold: Double = 0.5): Double ={
    val label = data.map(x => x._1)
    val predictL = data.map(x => DenseVector(x._2)).map{ x => this.fmModel.predictLabel(x, threshold)}
    FM.accuracy(label, predictL)
  }

  def accuracy(testSet: Array[(Int, Array[Double])], threshold: Double = 0.5): Double = {
    val label = testSet.map(x => x._1)
    val predictL = testSet.map(x => DenseVector(x._2)).map{ x => this.fmModel.predictLabel(x, threshold)}
    FM.accuracy(label, predictL)
  }

}
