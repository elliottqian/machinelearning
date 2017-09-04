package qian.node

import scala.collection.mutable

/**
  * 运算节点
  */
class MyNode extends Serializable {

  val input = new mutable.ArrayBuffer[MyNode]()
  var op: qian.op.Operation = _
  val const_attr = null
  var name: String = _

  def +(other: MyNode) {
    if (other.isInstanceOf[MyNode])
      null
  }

  def :*(other: MyNode): MyNode = {
    null
  }
}
