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

  var b: Double = scala.util.Random.nextDouble() - 0.5
  var w: DenseVector[Double] = DenseVector.rand[Double](featureSize) - 0.5
  var v: DenseMatrix[Double] = DenseMatrix.rand[Double](featureSize, k) - 0.5

  def getZ(x: DenseVector[Double]): Double = {
    var tempSum = 0.0
    for (i <- 0.until(this.featureSize))
      for (j <- (i + 1).until(this.featureSize))
        tempSum += x(i) * x(j) * (this.v(i,::) * this.v(j,::).t)
    tempSum + x.t * this.w + this.b
  }

  def predictLabel(x: DenseVector[Double]): Double = {
    val z = this.getZ(x)
    Activation.sigmoid(z)
  }

  def getGrad(x: DenseVector[Double], y: Int): (Double, DenseVector[Double], DenseMatrix[Double]) ={
    val yPredict = this.predictLabel(x)
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
    val yPredict = this.predictLabel(x)
    FM.getLoss(y, yPredict)
  }

  def updateParam(gradB: Double, gradW: DenseVector[Double], gradV: DenseMatrix[Double], stepSize: Double): Unit = {
    this.b -= gradB * stepSize
    this.w -= gradW * stepSize
    this.v -= gradV * stepSize
  }

  /**
    *def get_loss(self, x, y):
        y_predict = self.predict_label(x)
        return FM.log_loss(y, y_predict)
    * def get_grad_dz_dw(self, x):
        return x

    def get_grad_dz_dv(self, x):
        a = self.v * x.reshape((self.feature_size, 1))
        b = np.sum(a, 0)
        c = x.reshape((self.feature_size, 1)).dot(b.reshape((1, self.k)))
        x_2 = x * x
        t_ = self.v * x_2.reshape((self.feature_size, 1))
        r = c - t_
        return r
    * def update_parm(self, grad_b, grad_w, grad_v, step_size):
        self.b -= step_size * grad_b
        self.w -= step_size * grad_w
        self.v -= step_size * grad_v
    * @return
    *
    */



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
    if (y_predict <= 0.00001)
      newPredict = 0.00001
    else if (y_predict >= 0.99999)
      newPredict = 0.99999
    val l = y * scala.math.log(y_predict) + (1 - y) * scala.math.log(1 - y_predict)
    -l
  }


  def main(args: Array[String]): Unit = {

  }
}
