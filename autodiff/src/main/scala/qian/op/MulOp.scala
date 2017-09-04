package qian.op

import qian.node.MyNode

import scala.collection.mutable
import breeze.linalg.DenseVector

/**
  * 乘法操作, 向量相乘
  * 构造函数私有化
  */
class MulOp private extends Operation {

  /**
    * 计算内集
    * @param node
    * @param inputValArr
    * @return
    */
  def compute(node: MyNode, inputValArr: mutable.ArrayBuffer[DenseVector[Double]]): Double = {
    assert(inputValArr.length == 2)
    inputValArr(0).t * inputValArr(1)  // 两个向量相乘, 返回的是标量
  }

  /**
    * 关于MyNode符号重载
    * :* 表示求内集
    * *  表示element wise 相乘
    * @param node
    * @param outputGrad
    * @return
    */

  def gradient(node: MyNode, outputGrad: MyNode): mutable.ArrayBuffer[MyNode] = {
    val r = new mutable.ArrayBuffer[MyNode]()
    r += node.input(1) :* outputGrad
    r += node.input(0) :* outputGrad
    r
  }

  def apply(nodeA: MyNode, nodeB: MyNode): MyNode = {
    val newNode = super.apply()
    newNode.op = this
    newNode.name = s"${nodeA.name} :* ${nodeB.name}"
    newNode
  }
}


object MulOp {

  var instance: MulOp = _

  def getInstance(): MulOp = {
    this.synchronized {
      if (instance == null){
        println("first create")
        instance = new MulOp()
      }
      instance
    }

  }

  def main(args: Array[String]): Unit = {
    val mlop = MulOp.getInstance()
    val mlop2 = MulOp.getInstance()
    println(mlop == mlop2)


  }
}