package qian.fm

import breeze.linalg.{*, DenseMatrix, DenseVector, sum => bSum, Axis}
import qian.Activation
/**
  * Created by qianwei on 2017/9/4.
  * [@spec(Double, Float)]
  *
  * 矩阵和向量
加：+    减：-   点乘：  :*   点除：:/        矩阵乘法: *  矩阵除法: /
Matrix(*, ::)+Vector 逐行
Matrix(::, *)+Vector 逐列
  */
class FM(featureSize: Int, k: Int) extends qian.Classification{

  scala.util.Random.setSeed(10)

  var b: Double = 0.6//scala.util.Random.nextDouble() - 0.5
  println(b)
  var w: DenseVector[Double] = DenseVector.rand[Double](featureSize) - 0.5  //DenseVector(0.1, 0.3, 0.5)
  var v: DenseMatrix[Double] = DenseMatrix.rand[Double](featureSize, k) - 0.5 // DenseMatrix((0.15, 0.2),(0.25, 0.3),(0.35, 0.4))//

  def getZ(x: DenseVector[Double]): Double = {
    var tempSum = 0.0
    for (i <- 0.until(this.featureSize))
      for (j <- (i + 1).until(this.featureSize))
        tempSum += x(i) * x(j) * (this.v(i,::) * this.v(j,::).t)
    tempSum + x.t * this.w + this.b
  }

  /**
    * 预测分数
    * @param x
    * @return
    */
  def predictScore(x: DenseVector[Double]): Double = {
    val z = this.getZ(x)
    Activation.sigmoid(z)
  }

  def predictLabel(x: DenseVector[Double], threshold: Double): Int = {
    val score = predictScore(x)
    if (score >= threshold)
      1
    else
      0
  }


  def getGrad(x: DenseVector[Double], y: Int): (Double, DenseVector[Double], DenseMatrix[Double]) ={
    val yPredict = this.predictScore(x)
    val factor = yPredict - y.toFloat
    val dyDb = factor * this.getGradDzDb()
    val dyDw = this.getGradDzDw(x) * factor
    val dyDv = this.getGradDzDv(x) * factor
    (dyDb, dyDw, dyDv)
  }

  def getGradDzDb(): Double = 1.0

  def getGradDzDw(x: DenseVector[Double]): DenseVector[Double] = x

  def getGradDzDv(x: DenseVector[Double]): DenseMatrix[Double] = {
    // v的每一列乘以x
    val a = this.v(::, *) :* x
    val b = bSum(a, Axis._0)
    val c = x * b
    val x_2 = x :* x
    val d = this.v(::, *) :* x_2
    c - d
  }

  def getLoss(x: DenseVector[Double], y: Int): Double ={
    val yPredict = this.predictScore(x)
    FM.getLoss(y, yPredict)
  }

  def updateParam(gradB: Double, gradW: DenseVector[Double], gradV: DenseMatrix[Double], stepSize: Double): Unit = {
    this.b = this.b - (gradB * stepSize)
    this.w = this.w - (gradW * stepSize)
    this.v = this.v - (gradV * stepSize)
  }

  def checkGradW(x: DenseVector[Double], y: Int, dw:Double, col: Int): Unit ={
    val f = this.getLoss(x, y)
    println("旧的损失: " + f)
    this.w(col) = this.w(col) + dw
    val newF = this.getLoss(x, y)
    println("新的损失: " + newF)
    val dlDw = (newF - f) / dw
    println("梯度: " + dlDw)
    this.w(col) = this.w(col) - dw
  }


  def checkGradV(x: DenseVector[Double], y: Int, dv:Double, row: Int, col: Int): Unit = {
    val f = this.getLoss(x, y)
    println("旧的损失: " + f)
    this.v(row, col) = this.v(row, col) + dv
    val newF = this.getLoss(x, y)
    println("新的损失: " + newF)
    val dlDv = (newF - f) / dv
    println("梯度: " + dlDv)
    this.v(row, col) = this.v(row, col) - dv
  }



  override def toString: String = {
    val bs = b.toString
    val ws = w.toString
    val vs = v.toString()
    "b:" + bs + "\n" + ws + "\n" + vs
  }


}

object FM{

  def getLoss(y:Double, y_predict: Double): Double = {
    var newPredict = y_predict
    if (y_predict <= 0.0001)
      newPredict = 0.00001
    else if (y_predict >= 0.9999)
      newPredict = 0.9999
    val l = y * scala.math.log(newPredict) + (1 - y) * scala.math.log(1 - newPredict)
    -l
  }

  def getInstance(): Unit ={

  }


  def main(args: Array[String]): Unit = {

  }


  def accuracy(label: Array[Int], predict: Array[Int]): Double ={
    val length = label.length
    val correctLength = label.zip(predict).count(x => x._1 == x._2)
    correctLength.toDouble / length.toDouble
  }
}
