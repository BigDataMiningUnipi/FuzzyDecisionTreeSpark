package iet.unipi.bigdatamining.classification.tree.model.binary

import org.apache.spark.Logging

/**
  * Splits applied to a feature
  * @param feature feature index
  * @param categoryIndexes the indexes of the value in the feature where
  *          category belongs to.
  * @param categories the value ids for the split of the left node.
  *          It's the unique id of the value (both categorical or
  *          fuzzy partition) for all feature
  */
class BinarySplit (
                    val feature: Int,
                    val categoryIndexes: List[Int],
                    val categories: List[Int]) extends Serializable with Logging {

  /**
    * Return a string with a summary of the split
    */
  override def toString: String = {
    s"Feature = $feature, " + s"categoryIndexes = $categoryIndexes, categories = $categories"
  }

}