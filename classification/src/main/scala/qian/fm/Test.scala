package qian.fm

import breeze.linalg.{*, DenseMatrix, DenseVector}
import breeze.linalg.{Axis, DenseMatrix, sum => bSum}

/**
  * Created by qianwei on 2017/9/5.
  */
object Test {
  def main(args: Array[String]): Unit = {
    println("随机数种子")
    scala.util.Random.setSeed(10)

    println("初始化一个标量")
    val x = scala.util.Random.nextDouble() - 0.5
    println(x)

    println("初始化一个矩阵")
    val w = DenseMatrix.rand[Double](3, 3)
    println(w)

    val fm = new FM(3, 2)
    println(fm)

    println("初始化一个稠密向量")
    val v1 = DenseVector[Double](1, 2, 3)

    println("w的0行")
    println(w(0,::))
    println("w的第1行")
    println(w(1,::))
    println("w的0行转置")
    println(w(1,::).t)
    println("w的第1行乘以w的第一列")
    println(w(1,::) * w(1,::).t)

    println("w的每一列乘以 v1向量")
    println(w(::, *))
    println(w(::, *) :* v1)

    println("w的每一行乘以 v1向量")
    println(w(*, ::):* v1)//


    println("向量和向量点乘")
    println(v1 :* v1)

    println("矩阵按照纵向求和, 数字是保留的维度, 表示第0维度保留")
    println(bSum(w, Axis._0))

    println("矩阵乘以数字")
    println(v1 * 3.0)

      //............
    println("array转换成向量")
    val arr = Array(1.0, 2.0, 3.0)
    val v2 = new DenseVector[Double](arr)
    println(v2)
  }
}
