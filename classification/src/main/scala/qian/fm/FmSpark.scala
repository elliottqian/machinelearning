package qian.fm

import org.apache.spark.{SparkConf, SparkContext}

/**
  * 在spark的每个worker执行梯度下降
  */
object FmSpark {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("FmSpark")
    val sc = new SparkContext(conf)

    val data = sc.textFile("D:\\NewCode\\machinelearning\\classification\\src\\main\\resources\\x")
      .filter(x => x.startsWith("0") || x.startsWith("1")).map{ l =>
      val lines = l.split("\t")
      val y = lines(0).toInt
      val x = lines(1).toCharArray.map(x => x.toString.toDouble)
      (y, x)
    }


    data.foreach(println)



  }
}
