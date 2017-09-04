package qian.op

import qian.node.MyNode

trait Operation extends Serializable {

    def apply(): MyNode = new MyNode()

//  def compute(node: MyNode, inputValArr: mutable.ArrayBuffer[Double]): Double
//
//  def gradient(node: MyNode, outputGrad:Double): mutable.ArrayBuffer[MyNode]
}