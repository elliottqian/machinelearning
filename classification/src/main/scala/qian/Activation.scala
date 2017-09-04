package qian

import breeze.linalg.DenseVector

/**
  * Created by qianwei on 2017/9/4.
  */
object Activation {

  def sigmoid(z: Double): Double = 1 / (1 + scala.math.exp(-z))

  def sigmoid(z: DenseVector[Double]) = null

}
