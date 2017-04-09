package iet.unipi.bigdatamining.classification.tree.model.multi

import iet.unipi.bigdatamining.classification.tree.model.PredictStats
import iet.unipi.bigdatamining.classification.tree.model.Node
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.Vector
import iet.unipi.bigdatamining.fuzzy.fuzzyset.FuzzySet
import iet.unipi.bigdatamining.fuzzy.tnorm.TNorm

/**
  * Node in a Fuzzy Multi Decision Tree.
  *
  * About node indexing:
  *   Nodes are indexed from 1. Node 1 is the root;
  *        for a partition with P fuzzy sets the nodes 2, 3, ..., and P+1
  *        are the first, the second, ..., and the last children.
  *   Node index 0 is not used.
  *
  * @param id integer node id, from 1
  * @param valueId id of the value of both categorical or continuous partitions.
  *                This is id identify unequivocally a specific value or fuzzy set
  *                among all input features
  * @param level of the node inside the tree (root has level 0)
  * @param impurity current node impurity
  * @param isLeaf whether the node is a leaf
  * @param featureHistory array of the featureId considered until this node
  * @param splitFeatureIndex split to calculate left and right nodes
  * @param gainStats information gain stats
  * @param predictStats store information for predicting value at the node
  * @param children array of children (the number depends on the partition)
  * @since 1.0
  */
class MultiNode (
                  val id: Int,
                  val valueId: Int,
                  val level: Int,
                  var impurity: Double,
                  var isLeaf: Boolean,
                  val featureHistory: Array[Int],
                  var splitFeatureIndex: Int,
                  var gainStats: Option[MultiInformationGainStats],
                  var predictStats: PredictStats,
                  var children: Array[Option[MultiNode]]) extends Node with Serializable with Logging {

  /**
    * Return a string with a summary of node.
    */
  override def toString: String = {
    val featureHistoryStr = featureHistory.mkString("[",",","]")
    s"id=$id, valueId = $valueId, level = $level, impurity = $impurity, " +
      s"isLeaf = $isLeaf, featureHistory = $featureHistoryStr " +
      s"predict = $predictStats, split = $splitFeatureIndex, stats = $gainStats"
  }

  def isRoot: Boolean = id == 1

  /**
    * Return a map where each key is the class label and the value is the
    * confidence value. The confidence value is computed as sum of the
    * activation degrees determined by any leaf node of the tree for that class.
    *  -> The class activation degree (ADm) is calculated as the product between the weight (wm)
    *  associated with the m-th class label (Cm) in the leaf node and the membership degree
    *  of the vector features to the leaf node.
    *    -> The weight wm associated with Cm is proportional to the number of training examples
    *    of that class in the node.  It is calculated as follow:
    *        wm = |SCm|/|S|
    *    where |S| is the number of examples that belong to the node and |SCm| is the number
    *    of examples in S with class Cm.
    *    -> The membership degree is calculated according the tNorm of the fuzzy decision tree model
    *    and considering the overall fuzzy set associated with the node. At the root node,
    *    we consider currMD equal to one.
    * Each activated leaf node produces a list of class activation degrees, which are summed up
    * to compute the confidence value for that class
    *
    * @param point vector to classify
    * @param tNorm function to be used
    * @param idToFeatureIdFuzzySet map from unique id to feature id and fuzzy set
    * @param currAD current membership degree
    * @return the map with the class label as key and the confidence value as value
    * @since 1.0
    */
  private def predictLabels(point: Vector,
                            tNorm: TNorm,
                            idToFeatureIdFuzzySet: Map[Int, (Int, FuzzySet)],
                            currAD: Double): Map[Int, Double] = {

    val mdFeatureValue =  if (!isRoot){
      val (featureId, fuzzySet) = idToFeatureIdFuzzySet.get(valueId).get
      fuzzySet.membershipDegree(point(featureId))
    } else {
      1D
    }

    if (mdFeatureValue != 0D){
      if (isLeaf) {
        predictStats.getWeightedMD.zipWithIndex.map{ case (wm, index) =>
          (index, wm*tNorm.calculate(currAD, mdFeatureValue))
        }.toMap
      } else {
        children.map { node =>
          node.get.predictLabels(point, tNorm, idToFeatureIdFuzzySet, tNorm.calculate(currAD, mdFeatureValue))
        }.foldLeft(Map.empty[Int, Double])((map1, map2) =>
          map1 ++ map2.map{ case (k,v) => k -> (v + map1.getOrElse(k,0D)) })
      }
    } else {
      Map.empty[Int, Double]
    }
  }

  /**
    * Predict value if node is not leaf
    *
    * @param point value
    * @return predicted value
    * @since 1.0
    */
  def predict(point: Vector,
              tNorm: TNorm,
              idToFeatureIdFuzzySet: Map[Int, (Int,FuzzySet)]): Map[Int, Double] = {
    predictLabels(point, tNorm, idToFeatureIdFuzzySet, 1D)
  }

  /**
    * Get the all the children of the node
    *
    * @return an array where each element is a Node (i.e. a Multi Node)
    */
  private[tree] def sons: Array[Node] = children.map(_.get)

  /**
    * Return the sum of the membership degrees
    * of all points in the dataset from the root to the node
    */
  private[tree] def fuzzyCardinality: Double = {
    predictStats.totalU
  }

  /**
    * Return the number of points in the dataset
    * that fall from the root to the node
    */
  private[tree] def cardinality: Double = {
    predictStats.totalFreq.toInt
  }

  /**
    * Get a deep copy of the subtree rooted at this node.
    *
    * @return a deep copy of the subtree rooted at this node.
    */
  def deepCopy: MultiNode = {
    val childrenCopy = new Array[Option[MultiNode]](children.length)
    children.indices.foreach { i =>
      val child = children(i)
      if (child.nonEmpty)
        childrenCopy(i) = Some(child.get.deepCopy)
      else
        childrenCopy(i) = None
    }
    new MultiNode(id, valueId, level, impurity, isLeaf, featureHistory.clone(), splitFeatureIndex,
      gainStats, predictStats, childrenCopy)
  }

  /**
    * Get the number of nodes in tree below this node, including leaf nodes.
    * E.g., if this is a leaf, returns 0.  If it has two children and both are leaves, returns 2
    *
    * @return the number of nodes in tree below this node, including leaf nodes.
    */
  private[tree] def numDescendants: Int = {
    if (predictStats.totalFreq <= 0)
      0
    else
      1 + children.map { node => if (node.nonEmpty || node.get.isLeaf) node.get.numDescendants else 0}.sum
  }

  /**
    * Get the number of leaves in tree below this node.
    * Leaf with 0 instances are not considered
    * E.g., if this is a leaf, returns 1.
    *
    * @return the number of leaves in tree below this node.
    */
  private[tree] def numLeaves: Int = {
    val leaf = if (isLeaf && predictStats.totalFreq > 0) 1 else 0
    leaf + children.map { node => if (node.nonEmpty) node.get.numLeaves else 0}.sum
  }

  /**
    * Get depth of tree from this node.
    * E.g.: Depth 0 means this is a leaf node.
    *
    * @return depth of tree from this node.
    */
  private[tree] def subtreeDepth: Int = if (isLeaf) {
    0
  } else {
    1 + children.map { node => if (node.nonEmpty) node.get.subtreeDepth else 0 }.max
  }

  /**
    * Get depth of the shortest branch from this node.
    *
    * @return depth of the shortest branch from this node.
    */
  private[tree] def minSubtreeDepth: Int = if (isLeaf) {
    0
  } else {
    1 + children.map { node => if (node.nonEmpty) node.get.minSubtreeDepth else 0 }.min
  }

  /**
    * Get length of each branch
    *
    * @param numPrefix The number of previous levels
    * @return an iterator where each element contains the depth of a specifc branch.
    *         Size of iterator depends on the number of leaves
    */
  private[tree] def depthBranches(numPrefix: Int = 0): Iterator[Int] = {
    if (isLeaf){
      if (predictStats.totalFreq > 0){
        List(numPrefix).toIterator
      } else {
        List.empty[Int].toIterator
      }
    } else {
      children.map{ node =>
        if (node.nonEmpty) node.get.depthBranches(numPrefix + 1) else List.empty[Int].toIterator
      }.filter { iterator => iterator.nonEmpty}
       .map { iterator => iterator.next()}.toIterator
    }
  }

  /**
    * Recursive print function.
    *
    * @param indentFactor The number of spaces to add to each level of indentation.
    * @return a string representation of the subtree
    */
  private[tree] def subtreeToString(indentFactor: Int = 0): String = {

    def splitToString(splitFeatureIndex: Int, index: Int): String = {
      s"$splitFeatureIndex = $index"
    }

    val prefix: String = "|   " * indentFactor
    if (isLeaf){
      s" : $predictStats"
    }else{
      (0 until children.length).map{ i =>
        if (children(i).isEmpty){
          ""
        }else{
          if (children(i).get.predictStats.totalFreq > 0){
            "\n" + prefix + splitToString(splitFeatureIndex/*.get*/, i) +
              children(i).get.subtreeToString(indentFactor + 1)
          }else{
            ""
          }
        }
      }.mkString
    }
  }

  /**
    * The method traverses (DFS, left to right) the subtree of this node.
    *
    * @return an iterator that traverses (DFS, left to right) the subtree of this node.
    */
  def subtreeIterator: Iterator[Node] = {
    Iterator.single(this) ++ children.flatMap{ child =>
      if (child.nonEmpty)
        child.get.subtreeIterator
      else
        Iterator.empty
    }.toIterator
  }

}

/**
  * Companion object of MultiNode
  */
private[tree] object MultiNode{

  /**
    * Return a node with the given node id (but nothing else set).
    *
    * @param id of the node
    * @return an empty Node, i.e. a node with the given node id (but nothing else set).
    */
  def emptyNode(id: Int): MultiNode = new MultiNode(id, -1, 0, -1D, false, Array.empty[Int], -1, None,
    PredictStats.empty, Array.empty[Option[MultiNode]])

  /**
    * Construct a node with nodeId, nodeLevel, impurity, isLeaf
    * and predictStats parameters.
    * This is used in `FuzzyDecisionTree.findBestSplits` to construct child nodes
    * after finding the best splits for parent nodes.
    * Other fields are set at next level.
    *
    * @param id integer node index between siblings, starting from 0
    * @param valueId integer node id, starting from -1
    * @param nodeLevel intger of node level in the tree
    * @param impurity current node impurity
    * @param isLeaf whether the node is a leaf
    * @param predictStats prediction information on labels at the node
    * @return new node instance
    */
  def apply(
             id: Int,
             valueId: Int,
             nodeLevel: Int,
             impurity: Double,
             isLeaf: Boolean,
             predictStats: PredictStats,
             featureHistory: Array[Int] = Array.empty[Int]): MultiNode = {
    new MultiNode(id, valueId, nodeLevel, impurity, isLeaf, featureHistory,
      -1, None, predictStats, Array.empty[Option[MultiNode]])
  }

  /**
    * Retrieve the maximum id from all nodes belonging of the subtree of the node
    *
    * @param node the first node of the subtree on which retrieve the maximum id
    * @return the maximum id of the subtree, starting from the node.
    */
  private[tree] def getMaxId(node: MultiNode): Int = {
    if (node.isLeaf || node.splitFeatureIndex < 0) node.id
    else
      node.children.map { child => if (child.nonEmpty) MultiNode.getMaxId(child.get)}
        .foldLeft(node.id){ case (id1, id2) =>
          id2 match {
            case value: Int => if (id1 > value) id1 else value
            case _ => id1
          }
        }
  }

}


