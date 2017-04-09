package iet.unipi.bigdatamining.classification.tree.model.binary

import iet.unipi.bigdatamining.classification.tree.model.PredictStats
import iet.unipi.bigdatamining.classification.tree.model.Node

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.Vector
import iet.unipi.bigdatamining.fuzzy.fuzzyset.FuzzySet
import iet.unipi.bigdatamining.fuzzy.tnorm.TNorm

/**
  * Node in a Fuzzy Binary Decision Tree.
  *
  * About node indexing:
  * Nodes are indexed from 1. Node 1 is the root;
  * for a partition with P fuzzy sets the nodes 2, 3, ..., and P+1
  * are the first, the second, ..., and the last children.
  * Node index 0 is not used.
  *
  * @param id            integer node id, from 1
  * @param impurity      current node impurity
  * @param isLeaf        whether the node is a leaf
  * @param split         split to calculate left and right nodes
  * @param splitsHistory array of the featureId considered until this node
  * @param gainStats     information gain stats
  * @param predictStats  store information for the prediction value at the node
  * @param leftChild     left child node
  * @param rightChild    right child node
  * @since 1.0
  */
class BinaryNode(val id: Int,
                 var impurity: Double,
                 var isLeaf: Boolean,
                 var split: Option[BinarySplit],
                 val splitsHistory: Map[Int, Set[Int]],
                 var gainStats: Option[BinaryInformationGainStats],
                 var predictStats: PredictStats,
                 var leftChild: Option[BinaryNode],
                 var rightChild: Option[BinaryNode]) extends Node with Serializable with Logging {

  /**
    * Return a string with a summary of node
    */
  override def toString: String = {
    s"id = $id, isLeaf = $isLeaf, predict = $predictStats, impurity = $impurity, " +
      s"splitsHistory = $splitsHistory, split = $split, stats = $gainStats"
  }

  /**
    * Check if this node is the root of the tree
    *
    * @return true if the current node is the root
    */
  def isRoot: Boolean = id == 1

  /**
    * Return a map where each key is the class label and the value is the
    * confidence value. The confidence value is computed as sum of the
    * activation degrees determined by any leaf node of the tree for that class.
    * -> The class activation degree (ADm) is calculated as the product between the weight (wm)
    * associated with the m-th class label (Cm) in the leaf node and the membership degree
    * of the vector features to the leaf node.
    * -> The weight wm associated with Cm is proportional to the number of training examples
    * of that class in the node.  It is calculated as follow:
    * wm = |SCm|/|S|
    * where |S| is the number of examples that belong to the node and |SCm| is the number
    * of examples in S with class Cm.
    * -> The membership degree is calculated according the tNorm of the fuzzy decision tree model
    * and considering the overall fuzzy set associated with the node. At the root node,
    * we consider currMD equal to one. If the membership degree for a given feature in the mdHistory
    * has been already computed (i.e. is different from 1 or 0), then the tNorm is not calculated again
    * Each activated leaf node produces a list of class activation degrees, which are summed up
    * to compute the confidence value for that class
    *
    * @param features              vector to classify
    * @param tNorm                 used for calculating membership degree
    * @param idToFeatureIdFuzzySet map from unique id to feature id and fuzzy set
    * @param currAD                current association degree
    * @param mdHistory             the history if the previous association degree
    * @return the map with the class label as key and the confidence value as value
    * @since 1.0
    */
  private def predictLabels(
                             features: Vector,
                             tNorm: TNorm,
                             idToFeatureIdFuzzySet: Map[Int, (Int, FuzzySet)],
                             currAD: Double,
                             mdHistory: Map[Int, Int]): List[Map[Int, Double]] = {
    if (isLeaf) {
      List(predictStats.getWeightedMD.zipWithIndex.map { case (wm, index) =>
        (index, wm * currAD) // calculate the confidence value
      }.toMap)
    } else {
      // Store the sum of the md of each fuzzy set for a given point
      // Note that categorical values are treated as Singleton Fuzzy Set
      var mdFeatureValue = 0D
      var nodeMDHistory = -1
      mdHistory.get(split.get.feature) match {
        case None => // feature has never been used from the root to this node
          split.get.categories.foreach { valueId =>
            val (featureId, fuzzySet) = idToFeatureIdFuzzySet.get(valueId).get
            val memDegree = fuzzySet.membershipDegree(features(featureId)) //Calculate membership degree
            mdFeatureValue += memDegree
            if (memDegree > 0)
              nodeMDHistory = valueId
          }
          var predictionsList = List.empty[Map[Int, Double]]
          if (mdFeatureValue > 0D) {
            // There is at least a fired fuzzy set
            if (mdFeatureValue == 1D) {
              // Exactly two fuzzy sets are fired (or only one activated in the core)
              // In this case only the left branch is activated
              predictionsList = predictionsList ++
                leftChild.get.predictLabels(features, tNorm, idToFeatureIdFuzzySet,
                  tNorm.calculate(currAD, mdFeatureValue), mdHistory)
            } else {
              // In this case both the branches are activated with different activation degree
              predictionsList = predictionsList ++
                leftChild.get.predictLabels(features, tNorm, idToFeatureIdFuzzySet,
                  tNorm.calculate(currAD, mdFeatureValue),
                  mdHistory + (split.get.feature -> nodeMDHistory)) ++
                rightChild.get.predictLabels(features, tNorm, idToFeatureIdFuzzySet,
                  tNorm.calculate(currAD, 1D - mdFeatureValue),
                  mdHistory + (split.get.feature -> -1))
            }
            predictionsList
          } else {
            // In this case only the right branch is activated
            predictionsList ++
              rightChild.get.predictLabels(features, tNorm, idToFeatureIdFuzzySet,
                tNorm.calculate(currAD, 1D), mdHistory)
          }
        case Some(valueId) => // feature has been used at least once from the root to this node
          if (valueId != -1) {
            // Case where both branches have been activated
            // We don't compute again the activation degree for not penalizing branch with the same features
            // We check only if the fuzzy set belongs to the left or to the right branch
            if (split.get.categories.contains(valueId)) {
              leftChild.get.predictLabels(features, tNorm, idToFeatureIdFuzzySet,
                currAD, mdHistory)
            } else {
              rightChild.get.predictLabels(features, tNorm, idToFeatureIdFuzzySet,
                currAD, mdHistory)
            }
          } else {
            // Case where only one branch has been activated so far
            // We compute the membership degree in a similar way of previous code
            split.get.categories.foreach { valueId =>
              val (featureId, fuzzySet) = idToFeatureIdFuzzySet.get(valueId).get
              val memDegree = fuzzySet.membershipDegree(features(featureId))
              mdFeatureValue += memDegree
              if (memDegree > 0)
                nodeMDHistory = valueId
            }
            if (mdFeatureValue > 0D) {
              leftChild.get.predictLabels(features, tNorm, idToFeatureIdFuzzySet,
                currAD, mdHistory + (split.get.feature -> nodeMDHistory))
            } else {
              rightChild.get.predictLabels(features, tNorm, idToFeatureIdFuzzySet,
                currAD, mdHistory)
            }
          }
      }

    }
  }

  /**
    * Predict value if node is not leaf
    *
    * @param features feature value
    * @return predicted value
    * @since 1.0
    */
  def predict(features: Vector,
              tNorm: TNorm,
              idToFeatureIdFuzzySet: Map[Int, (Int, FuzzySet)]): Map[Int, Double] = {
    predictLabels(features, tNorm, idToFeatureIdFuzzySet, 1D, Map.empty[Int, Int])
      .foldLeft(Map.empty[Int, Double])((map1, map2) =>
        map1 ++ map2.map { case (k, v) => k -> (v + map1.getOrElse(k, 0D)) })
  }

  /**
    * Get the two children of the node
    *
    * @return an array where each element is a Node (i.e. a Binary Node)
    */
  private[tree] def sons: Array[Node] = {
    val buffer = new ArrayBuffer[Node]
    if (leftChild.nonEmpty)
      buffer += leftChild.get
    if (rightChild.nonEmpty)
      buffer += rightChild.get
    buffer.toArray
  }

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
    * Get the number of leaves in tree below this node.
    * Leaf with 0 instances are not considered
    * E.g., if this node is a leaf, returns 1.
    *
    * @return the number the leaves of the tree
    */
  private[tree] def numLeaves: Int = {
    var leaf = if (isLeaf && predictStats.totalFreq > 0) 1 else 0
    if (leftChild.nonEmpty) {
      leaf += leftChild.get.numLeaves
    }
    if (rightChild.nonEmpty) {
      leaf += rightChild.get.numLeaves
    }
    leaf
  }

  /**
    * Get the all the children of the node
    *
    * @return an array where each element is a Node (i.e. a Multi Node)
    */
  private[tree] def deepCopy: BinaryNode = {
    val leftChildCopy = if (leftChild.isEmpty) None else Some(leftChild.get.deepCopy)
    val rightChildCopy = if (rightChild.isEmpty) None else Some(rightChild.get.deepCopy)
    new BinaryNode(id, impurity, isLeaf, split, splitsHistory.map(x => x),
      gainStats, predictStats, leftChildCopy, rightChildCopy)
  }

  /**
    * Get the number of nodes in tree below this node, including leaf nodes.
    * E.g. if this node is a leaf, returns 1.  If both children are leaves, returns 3.
    *
    * @return the number of children from this node to each leaf including the f
    */
  private[tree] def numDescendants: Int = if (isLeaf) 1
  else 1 + leftChild.get.numDescendants + rightChild.get.numDescendants

  /**
    * Get depth of tree from this node.
    * E.g.: Depth 0 means this is a leaf node.
    *
    * @return the maximum depth from this node to the leaves
    */
  private[tree] def subtreeDepth: Int = if (isLeaf) 0
  else 1 + math.max(leftChild.get.subtreeDepth, rightChild.get.subtreeDepth)

  /**
    * Get depth of the shortest branch from this node.
    *
    * @return the minimum depth from this node to the leaves
    */
  private[tree] def minSubtreeDepth: Int = if (isLeaf) 0
  else 1 + math.min(leftChild.get.minSubtreeDepth, rightChild.get.minSubtreeDepth)

  //depthBranches(0).min

  /**
    * Get length of each branch
    *
    * @param numPrefix The number of previous levels
    * @return an iterator where each element contains the depth of a specifc branch.
    *         Size of iterator depends on the number of leaves
    */
  private[tree] def depthBranches(numPrefix: Int = 0): Iterator[Int] = if (isLeaf) {
    List(numPrefix).toIterator
  } else {
    leftChild.get.depthBranches(numPrefix + 1) ++
      rightChild.get.depthBranches(numPrefix + 1)
  }

  /**
    * Recursive print function.
    *
    * @param indentFactor The number of spaces to add to each level of indentation.
    * @return a string representation of the subtree
    */
  private[tree] def subtreeToString(indentFactor: Int = 0): String = {

    def splitToString(split: BinarySplit, left: Boolean): String = {
      if (left)
        s"${split.feature} in ${split.categoryIndexes.mkString("[", ",", "]")}"
      else
        s"${split.feature} not in ${split.categoryIndexes.mkString("[", ",", "]")}"
    }

    val prefix: String = "\n" + ("|   " * indentFactor)
    if (isLeaf) {
      s" => $predictStats"
    } else {
      prefix + s"If ${splitToString(split.get, left = true)}" +
        leftChild.get.subtreeToString(indentFactor + 1) +
        prefix + s"Else ${splitToString(split.get, left = false)}" +
        rightChild.get.subtreeToString(indentFactor + 1)
    }
  }

  /**
    * The method traverses (DFS, left to right) the subtree of this node.
    *
    * @return an iterator that traverses (DFS, left to right) the subtree of this node.
    */
  private[tree] def subtreeIterator: Iterator[Node] = {
    Iterator.single(this) ++ leftChild.map(_.subtreeIterator).getOrElse(Iterator.empty) ++
      rightChild.map(_.subtreeIterator).getOrElse(Iterator.empty)
  }

}

/**
  * Companion object of BinaryNode
  */
private[tree] object BinaryNode {

  /**
    * Return a node with the given node id (but nothing else set).
    *
    * @param nodeIndex index of the node
    * @return an empty Node, i.e. a node with the given node id (but nothing else set).
    */
  def emptyNode(nodeIndex: Int): BinaryNode = new BinaryNode(nodeIndex, -1D, false, None,
    Map.empty[Int, Set[Int]], None, new PredictStats(Array.empty[Double], Array.empty[Double]), None, None)

  /**
    * Construct a BinaryNode with nodeId, nodeLevel, impurity, isLeaf
    * and predictStats parameters.
    * This is used in `FuzzyDecisionTree.findBestSplits` to construct child nodes
    * after finding the best splits for parent nodes.
    * Other fields are set at next level.
    *
    * @param id            integer node id, from -1
    * @param splitsHistory the history of the create split from the root to this node
    * @param impurity      current node impurity
    * @param isLeaf        whether the node is a leaf
    * @param predictStats  information on labels at the node
    * @return a new BinaryNode instance
    */
  def apply(
             id: Int,
             splitsHistory: Map[Int, Set[Int]],
             impurity: Double,
             isLeaf: Boolean,
             predictStats: PredictStats): BinaryNode = {
    new BinaryNode(id, impurity, isLeaf, None, splitsHistory,
      None, predictStats, None, None)
  }

  /**
    * Get the index of the left child of this node.
    *
    * @param nodeIndex index of the node
    * @return the index of the left child of this node.
    */
  def leftChildIndex(nodeIndex: Int): Int = nodeIndex << 1

  /**
    * Get the index of the right child of this node.
    *
    * @param nodeIndex index of the node
    * @return the index of the right child of this node.
    */
  def rightChildIndex(nodeIndex: Int): Int = (nodeIndex << 1) + 1

  /**
    * Get the parent index of the given node, or 0 if it is the root.
    *
    * @param nodeIndex index of the node
    * @return the index of the parent node
    */
  def parentIndex(nodeIndex: Int): Int = nodeIndex >> 1

  /**
    * Get the level of a tree which the given node is in.
    *
    * @param nodeIndex the index of the node
    * @return the level of a tree which the given node is in.
    */
  def indexToLevel(nodeIndex: Int): Int = if (nodeIndex == 0) {
    throw new IllegalArgumentException(s"0 is not a valid node index.")
  } else {
    java.lang.Integer.numberOfTrailingZeros(java.lang.Integer.highestOneBit(nodeIndex))
  }

  /**
    * Check if the node identified by its index is a left child.
    * Note: Returns false for the root.
    *
    * @param nodeIndex the index of the node
    * @return true if this is a left child, false otherwise (or root)
    */
  def isLeftChild(nodeIndex: Int): Boolean = nodeIndex > 1 && nodeIndex % 2 == 0

  /**
    * Get the maximum number of nodes which can be in the given level of the tree.
    *
    * @param level of tree (0 = root).
    * @return the maximum number of nodes which can be in the given level of the tree
    */
  def maxNodesInLevel(level: Int): Int = 1 << level

  /**
    * Get the index of the first node in the given level.
    *
    * @param level Level of tree (0 = root).
    * @return the index of the first node in the given level.
    */
  def startIndexInLevel(level: Int): Int = 1 << level

  /**
    * Traces down from a root node to get the node with the given node index.
    * This assumes the node exists
    *
    * @param nodeIndex the index of the node
    * @param rootNode  the root of the Node
    * @return the BinaryNode identified by the given node index
    */
  def getNode(nodeIndex: Int, rootNode: BinaryNode): BinaryNode = {
    var tmpNode: BinaryNode = rootNode
    var levelsToGo = indexToLevel(nodeIndex)
    while (levelsToGo > 0) {
      if ((nodeIndex & (1 << levelsToGo - 1)) == 0) {
        tmpNode = tmpNode.leftChild.get
      } else {
        tmpNode = tmpNode.rightChild.get
      }
      levelsToGo -= 1
    }
    tmpNode
  }

}