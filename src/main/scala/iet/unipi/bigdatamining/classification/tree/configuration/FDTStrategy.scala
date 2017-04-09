package iet.unipi.bigdatamining.classification.tree.configuration

import scala.beans.BeanProperty
import scala.collection.JavaConverters._
import scala.collection.immutable.Map

/**
  * The class contains the information about the parameters used
  * fo learning the Fuzzy Decision Tree (both binary or multi split).
  *
  * @param splitType name of the splitting strategy
  * @param impurity name of the fuzzy impurity
  * @param tNorm name of the t-norm
  * @param maxDepth maximum depth of Fuzzy Decision Tree
  * @param maxBins maximum number of bins for each feature
  * @param numClasses number of class labels. It can take values {0, ..., numClasses - 1}.
  * @param categoricalFeaturesInfo a map that contains the categorical feature indexes as key
  *                                and the number of categorical values as value of the map
  * @param thresholdsFeatureInfo a map that contains the continuous feature indexes as key
  *                                and the an array of points as value of the map.
  *                                Points represent the threshold on which build the fuzzy partition
  * @param minInstancesPerNode minimum number of examples to be inspected in the stop condition (default 1)
  * @param minFuzzyInstancesPerNode minimum node fuzzy cardinality to be inspected in the stop condition,
  *                                 where the <i>node fuzzy cardinality</i> is computed as the sum
  *                                 of the membership degrees of all points in the dataset
  *                                 from the root to the node
  * @param minImpurityRatioPerNode minimum ratio thresholds of impurity that must be true in each node
  *                                (default 0)
  * @param minInfoGain minimum information gain threshold that must be true in each node (default 0).
  * @param subsamplingRate ratio of subsampling (default 1, i.e. all dataset is considered)
  * @param maxMemoryInMB maximum memory (in MB) to be used for each iteration (default 256MB)
  */
private[tree] class FDTStrategy(
   @BeanProperty var splitType: String,
   @BeanProperty var impurity: String,
   @BeanProperty var tNorm: String,
   @BeanProperty var maxDepth: Int,
   @BeanProperty var maxBins: Int,
   @BeanProperty var numClasses: Int,
   @BeanProperty var categoricalFeaturesInfo: Map[Int, Int] = Map.empty[Int, Int],
   @BeanProperty var thresholdsFeatureInfo: Map[Int, Array[Double]] = Map.empty[Int, Array[Double]],
   @BeanProperty var minInstancesPerNode: Int = 1,
   @BeanProperty var minFuzzyInstancesPerNode: Double = 0D,
   @BeanProperty var minImpurityRatioPerNode: Double = 1D,
   @BeanProperty var minInfoGain: Double = 0.000001,
   @BeanProperty var subsamplingRate: Double = 1D,
   @BeanProperty var maxMemoryInMB: Int = 256) extends Serializable {

  /**
   * Sets categoricalFeaturesInfo using a Java Map.
   *
   * @since 1.0
   */
  def setCategoricalFeaturesInfo(
      categoricalFeaturesInfo: java.util.Map[java.lang.Integer, java.lang.Integer]): Unit = {
    this.categoricalFeaturesInfo =
      categoricalFeaturesInfo.asInstanceOf[java.util.Map[Int, Int]].asScala.toMap
  }

  /**
   * Sets featurePartition using a Java Map.
   *
   * @since 1.0
   */
  def setThresholdsFeatureInfo(
      thresholdsFeatureInfo: java.util.Map[java.lang.Integer, java.util.List[java.lang.Double]]): Unit = {
    this.thresholdsFeatureInfo =
    thresholdsFeatureInfo.asInstanceOf[java.util.Map[Int, Array[Double]]].asScala.toMap
  }

  /**
   * Check validity of parameters.
   * Throws exception if invalid.
   */
  private[tree] def assertValid(): Unit = {
    require(numClasses >= 2,
      s"Fuzzy Decision Tree for Classification must have number of class label greater than 1, " +
      s"but numClasses = $numClasses.")
    require(Set("binary_split", "multi_split").contains(splitType.toLowerCase),
      s"Fuzzy Decision Tree Strategy given invalid type of splitting branch: $splitType. " +
      s"Valid settings are: 'binary_split' or 'multi_split'")
    require(Set("fuzzy_entropy").contains(impurity.toLowerCase),
      s"Fuzzy DecisionTree Strategy given invalid impurity for Classification: $impurity. " +
      s"Valid settings are: fuzzy_entropy")
    require(Set("min", "product").contains(tNorm.toLowerCase),
      s"Fuzzy Decision Tree Strategy given invalid t-norm for Classification: $tNorm. " +
      s"Valid settings are: min, product")
    require(!(categoricalFeaturesInfo.keySet exists thresholdsFeatureInfo.keySet),
      s"The key sets between categoricalFeatureInfo and thresholdsFeatureInfo should be disjoint! " +
      s"The same feature can't be categorical and continuous at the same time!" +
      s"categoricalFeaturesInfo keys: ${categoricalFeaturesInfo.keySet.toList.sorted}\n" +
      s"thresholdsFeatureInfo keys: ${thresholdsFeatureInfo.keySet.toList.sorted}")
    require(maxDepth > 0, s"Fuzzy Decision Tree Strategy given invalid maxDepth parameter: $maxDepth. " +
      s"Valid values are integers > 0.")
    require(maxBins >= 2, s"Fuzzy Decision Tree Strategy given invalid maxBins parameter: $maxBins. " +
      s"Valid values are integers >= 2.")
    require(minInstancesPerNode >= 1,
      s"Fuzzy Decision Tree Strategy requires minInstancesPerNode >= 1 but was given $minInstancesPerNode")
    require(minFuzzyInstancesPerNode >= 0,
      s"Fuzzy Decision Tree Strategy requires minFuzzyInstancesPerNode >= 0 but was given $minFuzzyInstancesPerNode")
    require(minImpurityRatioPerNode > 0 && minImpurityRatioPerNode <= 1D,
      s"Fuzzy Decision Tree Strategy requires minImpurityRatioPerNode greater than 0 and less or equal then 1.0")
    require(maxMemoryInMB <= 10240,
      s"Fuzzy Decision Tree Strategy requires maxMemoryInMB <= 10240, but was given $maxMemoryInMB")
    require(subsamplingRate > 0D && subsamplingRate <= 1D,
      s"Fuzzy Decision Tree Strategy requires subsamplingRate <=1 and >0, but was given " +
      s"$subsamplingRate")
  }

  /**
   * Returns a shallow copy of this instance.
   *
   * @since 1.0
   */
  def copy: FDTStrategy = {
    new FDTStrategy(splitType, impurity, tNorm, maxDepth, maxBins, numClasses, categoricalFeaturesInfo,
      thresholdsFeatureInfo, minInstancesPerNode, minImpurityRatioPerNode, minFuzzyInstancesPerNode,
      minInfoGain, subsamplingRate, maxMemoryInMB)
  }

}

object FDTStrategy {

  /**
   * Construct a default set of parameters for Fuzzy Decision Tree (both binary or multi branches)
   *
   * @since 1.0
   */
  def defaultStrategy(): FDTStrategy = {
    new FDTStrategy(splitType = "binary_split", impurity = "fuzzy_entropy", tNorm = "product",
            maxDepth = 5, maxBins = 32, numClasses = 2)
  }  
  
}
