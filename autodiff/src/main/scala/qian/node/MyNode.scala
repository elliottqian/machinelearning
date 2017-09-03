package qian.node

import scala.collection.mutable

/**
  * 运算节点
  */
class MyNode extends Serializable{
  val input = new mutable.ArrayBuffer[MyNode]()
  val op = null
  val const_attr = null
  var name = ""

  def +(other: MyNode) {
    if (other.isInstanceOf[MyNode])
      null
  }

}
