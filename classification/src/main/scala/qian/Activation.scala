package qian

/**
  * Created by qianwei on 2017/9/4.
  */
object Activation {

  def sigmoid(z: Double): Unit = 1 / (1 + scala.math.exp(-z))

}
