package qian.fm

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Created by qianwei on 2017/9/5.
  * 训练FM 模型
  */
class TrainFm extends Serializable{

  var fmModel:FM = null
  var data: Array[(Int, Array[Double])] = _
  var dim:Int = 0
  var k: Int = 0

  def setData(data: Array[(Int, Array[Double])]): Unit ={
    this.data = data
  }

  def setDim(): Unit ={
    this.dim = this.data(0)._2.length
  }

  def initFmModel(): Unit = {
    this.fmModel = new FM(this.dim, this.k)
  }

  def train(itNum: Int, stepSize: Double): Unit = {
    for (i <- 0.until(itNum)) {
      val (sumGradB, sumGradW, sumGradV) = this.trainStep()
      this.fmModel.updateParam(sumGradB, sumGradW, sumGradV, stepSize)
    }
  }

  def trainStep(): (Double, DenseVector[Double], DenseMatrix[Double]) ={
    var grad_b = 0.0
    var grad_w = DenseVector.zeros[Double](this.dim)
    var grad_v = DenseMatrix.zeros[Double](this.dim, this.k)
    for (oneRow <- data) {
      val tempX = new DenseVector[Double](oneRow._2)
      val (b, w, v) = this.fmModel.getGrad(tempX, oneRow._1)
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
    }
    loss
  }
}
