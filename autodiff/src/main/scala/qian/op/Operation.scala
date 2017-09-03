package qian.op

import qian.node.MyNode
import scala.collection.mutable

trait Operation extends Serializable {

  def compute(node: MyNode, inputValArr: mutable.ArrayBuffer[Double]): Double

  def gradient(node: MyNode, outputGrad:Double): mutable.ArrayBuffer[MyNode]


}

/**
  *     """Op represents operations performed on nodes."""
    def __call__(self):
        """Create a new node and associate the op object with the node.

        Returns
        -------
        The new node object.
        """
        new_node = Node()
        new_node.op = self
        return new_node

    def compute(self, node, input_vals):
        """Given values of input nodes, compute the output value.

        Parameters
        ----------
        node: node that performs the compute.
        input_vals: values of input nodes.

        Returns
        -------
        An output value of the node.
        """
        assert False, "Implemented in subclass"

    def gradient(self, node, output_grad):
        """Given value of output gradient, compute gradient contributions to each input node.

        Parameters
        ----------
        node: node that performs the gradient.
        output_grad: value of output gradient summed from children nodes' contributions

        Returns
        -------
        A list of gradient contributions to each input node respectively.
        """
        assert False, "Implemented in subclass"
  *
  *
  *
  */