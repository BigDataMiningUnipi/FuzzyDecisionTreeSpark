package iet.unipi.bigdatamining.classification.tree.model

import iet.unipi.bigdatamining.fuzzy.tnorm.TNorm
import iet.unipi.bigdatamining.fuzzy.fuzzyset.FuzzySet
import org.apache.spark.mllib.linalg.Vector

/**
  * Trait that defines the common methods of nodes of Fuzzy Decision Tree.
  * Both Binary node used in Fuzzy Binary Decision Tree and
  * Multi node used in Fuzzy Multi Decision Tree implement such a trait
  */
trait Node extends Serializable {
  
  /**
   * Predict value if node is not leaf
   */
  def predict(
      features: Vector, 
      Norm: TNorm, 
      idToFeatureIdFuzzySet: Map[Int, (Int, FuzzySet)]): Map[Int, Double]

  /**
   * True if node is the root of the tree, false otherwise
   */
  def isRoot: Boolean

  /**
   * Return children of the node as array
   */
  private[tree] def sons: Array[Node]

  /**
    * Return the sum of the sum of the membership degrees
    * of all points in the dataset from the root to the node
    */
  private[tree] def fuzzyCardinality: Double

  /**
    * Return the number of points in the dataset
    * that fall from the root to the node
    */
  private[tree] def cardinality: Double

  /**
   * Returns a deep copy of the subtree rooted at this node.
   */
  private[tree] def deepCopy: Node
    
  /**
   * Get the number of nodes in tree below this node, including leaf nodes.
   */
  private[tree] def numDescendants: Int

  /**
   * Get the number of leaves in tree below this node.
   * Leaf with 0 instances are not considered.
   * E.g., if this node is a leaf, returns 1.
   */
  private[tree] def numLeaves: Int

  /**
   * Get depth of tree from this node.
   */
  private[tree] def subtreeDepth: Int
  
  /**
   * Get depth of the shortest branch from this node.
   */
  private[tree] def minSubtreeDepth: Int
  
  /**
   * Get length of each branch
   * @param numPrefix the length of the prefix.
   *      0 in case of root.
   */
  private[tree] def depthBranches(numPrefix: Int = 0): Iterator[Int]
  
  /**
   * Recursive print function.
   */
  private[tree] def subtreeToString(indentFactor: Int = 0): String
  
  /** 
   *  Returns an iterator that traverses (DFS, left to right) the subtree of this node. 
   */
  private[tree] def subtreeIterator: Iterator[Node]
  
}