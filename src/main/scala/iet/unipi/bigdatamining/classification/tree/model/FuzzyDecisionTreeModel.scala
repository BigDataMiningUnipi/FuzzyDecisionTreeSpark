package iet.unipi.bigdatamining.classification.tree.model

import org.apache.spark.rdd.RDD
import org.apache.spark.api.java.JavaRDD

import org.apache.spark.mllib.linalg.Vector

import iet.unipi.bigdatamining.fuzzy.tnorm.TNorm
import iet.unipi.bigdatamining.fuzzy.fuzzyset.FuzzySet

/**
  * Fuzzy Decision tree model for classification.
  * This model stores the fuzzy decision tree structure and parameters.
  *
  * @param topNode root node
  * @param tNorm function that must be used
  * @param idToFeatureIdFuzzySet map from unique id to feature index and fuzzy set
  * @since 1.0
  */
// TODO: create methods for saving and loading the model
class FuzzyDecisionTreeModel (
                               val topNode: Node,
                               val tNorm: TNorm,
                               val idToFeatureIdFuzzySet: Map[Int, (Int, FuzzySet)]) extends Serializable {

  /**
    * Predict values for a single data point using the model trained.
    *
    * @param features array representing a single data point
    * @return prediction of the trained model
    * @since 1.0
    */
  def predict(features: Vector): Map[Int, Double] = {
    topNode.predict(features, tNorm, idToFeatureIdFuzzySet)
  }

  /**
    * Predict values with the highest vote for a single data point
    * using the trained model.
    *
    * @param point array representing a single data point
    * @return prediction of the trained model
    * @since 1.0
    */
  def predictByMaxValue(point: Vector): Double = {
    val predictions = topNode.predict(point, tNorm, idToFeatureIdFuzzySet)
    if (predictions.isEmpty){
      -1D
    } else{
      predictions.keysIterator.reduceLeft{(x,y) =>
        if (predictions(x) > predictions(y)) x else y
      }.toDouble
    }
  }

  /**
    * Predict values for a given distributed dataset using the trained model.
    *
    * @param data RDD representing data points to predict
    * @return RDD of predictions for each of the given data points
    * @since 1.0
    */
  def predict(data: RDD[Vector]): RDD[Map[Int, Double]] = {
    val bcModel = data.context.broadcast(this)
    data.mapPartitions{ points =>
      val model = bcModel.value
      points.map(x => model.predict(x))
    }
  }

  /**
    * Predict values for a given distributed dataset using the trained model.
    *
    * @param data a JavaRDD representing data points to be predicted
    * @return JavaRDD of predictions for each of the given data points
    *
    * @since 1.0
    */
  def predict(data: JavaRDD[Vector]): JavaRDD[Double] = {
    predict(data.rdd)
  }

  /**
    * Get number of nodes in tree
    * and all nodes that doesn't have instances belongs to.
    *
    * @since 1.0
    */
  def numNodes: Int = 1 + topNode.sons.map(_.numDescendants).sum

  /**
    * Get number of nodes in leaves.
    * @since 1.0
    */
  def numLeaves: Int = topNode.numLeaves

  /**
    * Get depth of tree.
    * E.g.: Depth 0 means 1 leaf node.  Depth 1 means 1 internal node and n leaf nodes.
    * @since 1.0
    */
  def depth: Int = topNode.subtreeDepth

  /**
    * Get the depth of the shortest branch
    */
  def minDepth: Int = topNode.minSubtreeDepth

  /**
    * Get the average depth of all branches in the tree
    */
  def averageDepth: Double = {
    val depthBranches = topNode.depthBranches(0).toArray
    depthBranches.sum.toDouble / depthBranches.length.toDouble
  }

  /**
    * Return a string with a summary of the model.
    */
  override def toString: String = {
    s"Fuzzy Decision Tree Model classifier of depth $depth with $numNodes nodes and $numLeaves leaves"
  }

  /**
    * The string representing the fuzzy decision tree model.
    * For debugging only porpoise.
    */
  def toDebugString: String = {
    toString() + "\n\tMinDepth=" + minDepth +
      " AverageDepth=" + averageDepth + "\n" +
      topNode.subtreeToString(0)
  }

}