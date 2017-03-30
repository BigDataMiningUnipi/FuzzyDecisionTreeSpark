package iet.unipi.bigdatamining.classification.tree.impl

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import iet.unipi.bigdatamining.fuzzy.fuzzyset.FuzzySet

/**
 * Internal representation of LabeledPoint for FuzzyDecisionTree:
 *  (a) Continuous features are binned based on fuzzy sets partition 
 *  (b) Categorical features are binned based on feature values.
 *
 * @param label  Label from LabeledPoint (the class)
 * @param binnedFeatures  Binned feature values.
 *             Same length as LabeledPoint.features, but values are bin indices.
 *             For continuous features, each point belongs to more fuzzy sets 
 *             (two in case of strong partition). For each point, we store
 *             an array of tuple where each tuple contains the index of fuzzy set
 *             in the partition on which fuzzy set is defined on
 *             and its uAi (membership degree)
 *             For categorical features, the array has only one element and we store
 *             the value of the feature that is an indices of all possible values for the 
 *             categorical feature (in this case membership degrees have no meaning and
 *             the associate double will be stored with value 1.0).
 *             Practically, categorical features are turned into a partitions of singleton features
 *             where a categorical value defines the core of the corresponding singleton fuzzy set.
 */
private[tree] class TreePoint(val label: Double, val binnedFeatures: Array[Array[(Int,Double)]])
  extends Serializable {

  /**
    * The string representing the tree point.
    * For debugging only porpoise.
    */
  def toDebugString: String = {
     val sb = new StringBuilder("[")
     binnedFeatures.foreach{ binnedFeature =>
       sb.append("{") 
       binnedFeature.foreach{ case (j, uA) =>
         sb.append("(" + j + ":" + uA + ")")
       }
       sb.append("}") 
     }
     sb.append("] -> " + label)
     sb.toString()
  }

}

private[tree] object TreePoint {
  
  /**
   * Convert an input dataset into its TreePoint representation,
   * binning feature values in preparation for FuzzyDecisionTree training.
   * @param input     Input dataset.
   * @param metadata  Learning and dataset metadata
   * @return  TreePoint dataset representation
   */
  def convertToTreeRDD(
      input: RDD[LabeledPoint],
      metadata: FuzzyDecisionTreeMetadata): RDD[TreePoint] = {
    // Construct arrays for categoricalFeatureArity for efficiency in the inner loop.
    val categoricalFeaturesArity: Array[Int] = new Array[Int](metadata.numFeatures)
    (0 until metadata.numFeatures).foreach { featureIndex =>
      categoricalFeaturesArity(featureIndex) = metadata.categoricalFeaturesArity.getOrElse(featureIndex, 0)
    }
    input.map { x =>
      TreePoint.labeledPointToTreePoint(x, categoricalFeaturesArity, metadata.fuzzySetFeatures)
    }
  }
  
  /**
   * Convert one LabeledPoint into its TreePoint representation.
   * @param labeledPoint  the labeled point to convert.
   * @param featureArity  Array indexed by feature, with value 0 for continuous and numCategories
   *                      for categorical features.
   * @param fuzzySetFeatures  fuzzy sets defined over each feature
   */
  private def labeledPointToTreePoint(
      labeledPoint: LabeledPoint,
      featureArity: Array[Int],
      fuzzySetFeatures: Map[Int, Array[FuzzySet]]): TreePoint = {
    val numFeatures = labeledPoint.features.size
    val arr = new Array[Array[(Int, Double)]](numFeatures)
    (0 until numFeatures).foreach{ featureIndex =>
      if (featureArity(featureIndex) == 0) {
        // Continuous feature
        arr(featureIndex) = findBinsForContinuousFeature(featureIndex,
                                labeledPoint.features(featureIndex), fuzzySetFeatures(featureIndex))       
      } else {
        // Categorical feature
        arr(featureIndex) = findBinsForCategoricalFeature(featureIndex,
                                labeledPoint.features(featureIndex), featureArity(featureIndex))
      }
    }
    
    new TreePoint(labeledPoint.label, arr)
  }
  
  /**
   * Find bin for one (labeledPoint, feature).
   *
    * @param featureIndex  index of the feature.
    * @param featureValue value of the feature
    * @param fuzzySetFeatures  array of fuzzy sets.
    * @return an array of tuple where each element contains the index of fuzzy set
    *         and the corresponding membership degree for a given feature value.
    */
  private def findBinsForContinuousFeature(
      featureIndex: Int,
      featureValue: Double,
      fuzzySetFeatures: Array[FuzzySet]): Array[(Int, Double)] = {
    
    /**
     * Binary search helper method for continuous feature that
     * from a given value find the index of bins.
     * Code similar to the classic binary search implemented in Java.
     *
     * @return a tuple that contains the indexes bins on which the corresponding
      *         membership degree is greater then 0
     */
    def binarySearchValueToBinIndexes: (Int, Int) = {
      var left = 0
      var right = fuzzySetFeatures.length - 1
      while (left <= right) {
        val mid = left + (right - left) / 2
        val fuzzySet = fuzzySetFeatures(mid)
        if (fuzzySet.isInSupport(featureValue)) {
          if (mid > 0 && fuzzySetFeatures(mid-1).isInSupport(featureValue))
            return (mid-1, mid)
          if (mid < (fuzzySetFeatures.length-1) && fuzzySetFeatures(mid+1).isInSupport(featureValue))
            return (mid, mid+1)
          return (mid, -1)
        } else if (fuzzySet.left >= featureValue) {
          right = mid - 1
        } else {
          left = mid + 1
        }
      }
      (-1,-1)
    }
    
    // Perform binary search for retrieving bin for continuous features.
    // Note that we are assuming to have a strong fuzzy partition,
    // where a point belongs only to two fuzzy sets at most
    val binIndexes = binarySearchValueToBinIndexes
    if (binIndexes._1 == -1 && binIndexes._2 == -1) {
      throw new RuntimeException("No fuzzy sets have been found for continuous feature value." +
        " This error can occur when given invalid data values (such as NaN)." +
        s" Feature index: $featureIndex.  Feature value: $featureValue")
    }
   
    val buffer = ArrayBuffer.empty[(Int, Double)]
    buffer += ((binIndexes._1, fuzzySetFeatures(binIndexes._1).membershipDegree(featureValue)))
    if (binIndexes._2 != -1)
      buffer += ((binIndexes._2, 1D - buffer(0)._2))
      
    buffer.toArray
  }
  
  /**
   * Find bin for one (labeledPoint, feature) where feature is categorical.
   *
   * @param featureIndex  index of the feature.
   * @param featureValue value of the feature
   * @param featureArity  0 for continuous features; number of categories for categorical features.
   * @return an array of tuple that contains exactly one element where the first value is
   *         the index of the category in the categorical feature and the second value is always 1.
   */
  private def findBinsForCategoricalFeature(
      featureIndex: Int,
      featureValue: Double,
      featureArity: Int): Array[(Int, Double)] = {  
    // Categorical feature bins are indexed by feature values.
    if (featureValue < 0 || featureValue >= featureArity) {
      throw new IllegalArgumentException(
        s"Fuzzy Decision Tree given invalid data:" +
         s" Feature $featureIndex is categorical with values in" +
         s" {0,...,${featureArity - 1}," +
         s" but a data point gives it value $featureValue.\n")
    }
    
    Array[(Int, Double)]((featureValue.toInt, 1D))
  }
  
}
