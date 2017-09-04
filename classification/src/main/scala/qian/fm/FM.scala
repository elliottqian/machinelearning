package qian.fm

import breeze.linalg.{*, DenseMatrix, DenseVector}
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

  def getGradDzDb(): Double = 1.0

  def getGradDzDw(x: DenseVector[Double]): DenseVector[Double] = x

  def getGradDzDv(x: DenseVector[Double]): DenseMatrix[Double] = {
    null
  }

  /**
    *
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
    */


  override def toString: String = {
    val bs = b.toString
    val ws = w.toString
    val vs = v.toString()
    "b:" + bs + "\n" + ws + "\n" + vs
  }


}

object FM{
  def main(args: Array[String]): Unit = {
    scala.util.Random.setSeed(10)
    val x = scala.util.Random.nextDouble() - 0.5
    println(x)

    val w = DenseMatrix.rand[Double](3, 3)
    println(w)

    val fm = new FM(3, 2)
    println(fm)

    val v1 = DenseVector[Double](1, 2, 3)
    println(w(0,::))
    println(w(1,::))
    println(w(1,::).t)
    println(w(1,::) * w(1,::).t)

    w(::, *) * v1
    println(w(::, *) * v1.t)
  }
}
