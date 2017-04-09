package iet.unipi.bigdatamining.classification.tree.impl


import scala.collection.mutable
import scala.collection.immutable.SortedMap
import org.apache.spark.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import iet.unipi.bigdatamining.classification.tree.configuration.{FDTStrategy, SplitType}
import iet.unipi.bigdatamining.fuzzy.tnorm.{TNorm, TNorms}
import iet.unipi.bigdatamining.fuzzy.fuzzyset.FuzzySet
import iet.unipi.bigdatamining.fuzzy.fuzzyset.TriangularFuzzySet
import iet.unipi.bigdatamining.classification.tree.impurity.{FuzzyImpurities, FuzzyImpurity}
import iet.unipi.bigdatamining.classification.tree.configuration.SplitType.{Binary, Multi}

/**
 * The class build the metadata for both Learning and dataset of FuzzyDecisionTree (both binary or multi split).
 *
 * @param numFeatures number of features
 * @param numExamples number of examples in the dataset
 * @param numClasses  number of class labels. It can take values {0, ..., numClasses - 1}.
 * @param categoricalFeaturesArity  Map: categorical feature index (key) --> arity (value).
 *                      Namely, the feature takes values in {0, ..., arity - 1}.
 * @param fuzzySetFeatures Map: continuous feature index (keY) --> array of fuzzy set (value).
 * @param numFeatureBin Map: feature index (key) --> number of bins (value).
 *                      Both continuous or categorical features.
 * @param unorderedFeatures a set containing the index of the unordered features.
 * @param featureIndexAndIndexInPartitionToId a map that contains as key the feature index and
  *                                            the index of the fuzzy set in such a feature and a unique
  *                                            id as value, namely (featureIndex, valueIndex) --> unique id
 * @param idToFeatureIndexAndIndexInPartition the opposite of featureIndexAndIndexInPartitionToId
 * @param impurity the fuzzy impurity used for learning the Fuzzy Decision Tree
 * @param tNorm the tNorm used for learning the Fuzzy Decision Tree
 * @param maxDepth maximum depth of Fuzzy Decision Tree
 * @param maxBins maximum number of bins for each feature
 * @param minInstancesPerNode minimum number of examples to be inspected in the stop condition (default 1)
 * @param minFuzzyInstancesPerNode minimum node fuzzy cardinality to be inspected in the stop condition,
 *                                 where the <i>node fuzzy cardinality</i> is computed as the sum
 *                                 of the membership degrees of all points in the dataset
 *                                 from the root to the node
 * @param minImpurityRatioPerNode minimum ratio threshold  of impurity that must be true in each node
 * @param minInfoGain minimum information gain threshold that must be true in each node.
 */
private[tree] class FuzzyDecisionTreeMetadata(
    val numFeatures: Int,
    val numExamples: Long,
    val numClasses: Int,
    val categoricalFeaturesArity: Map[Int, Int],
    val fuzzySetFeatures: Map[Int, Array[FuzzySet]] = Map[Int, Array[FuzzySet]](),
    val numFeatureBin: Map[Int, Int],
    val unorderedFeatures: Set[Int],
    val featureIndexAndIndexInPartitionToId: Map[(Int, Int), Int],
    val idToFeatureIndexAndIndexInPartition: Map[Int, (Int, Int)],
    val impurity: FuzzyImpurity,
    val tNorm: TNorm,
    val maxDepth: Int,
    val maxBins: Int,
    val minInstancesPerNode: Int,
    val minFuzzyInstancesPerNode: Double,
    val minImpurityRatioPerNode: Double,
    val minInfoGain: Double) extends Serializable {

  /**
    * Check if feature is unordered.
    * Used only for Fuzzy Binary Decision Trees.
    *
    * @param featureIndex the index of the feature to be checked
    * @return true if feature is unordered, false otherwise
    */
  def isUnordered(featureIndex: Int): Boolean = unorderedFeatures.contains(featureIndex)

  /**
    * Check if the problem is a multi classification problem.
    *
    * @return true if the problem is a multi classification problem, false otherwise
    */
  def isMultiClass: Boolean = numClasses > 2

  /**
    * Check if a given feature is categorical
    *
    * @param featureIndex the index of the feature to be checked
    * @return true if the feature is continuous, false otherwise
    */
  def isCategorical(featureIndex: Int): Boolean = categoricalFeaturesArity.contains(featureIndex)

  /**
    * Check if a given feature is continuous
    *
    * @param featureIndex the index of the feature to be checked
    * @return true if the feature is continuous, false otherwise
    */
  def isContinuous(featureIndex: Int): Boolean = fuzzySetFeatures.contains(featureIndex)

  /**
   * The number of values for a given feature. For continuous features is the number of fuzzy sets
   * defined in the partition, for categorical features is the arity of the feature.
    *
   * @param featureIndex the index of the feature to be checked
   * @return the number of values for a given feature
   */
  def numCategories(featureIndex: Int): Int = if (isContinuous(featureIndex)){
    fuzzySetFeatures(featureIndex).length
  }else{
    categoricalFeaturesArity(featureIndex)
  }
 
  /**
   * Number of splits for a given feature.
   * For unordered features, there are 2 bins per split.
   * For ordered features, there is 1 more bin than split.
   * Used only for Fuzzy Binary Decision Trees.
   *
   * @param featureIndex the index of the feature to be checked
   * @return the number of split for the given feature
   */
  def numSplits(featureIndex: Int): Int = if (isUnordered(featureIndex)) {
    numFeatureBin(featureIndex) >> 1
  } else {
    numFeatureBin(featureIndex) - 1
  }

}

/**
  * Object that implement some useful method for creating metadata of Fuzzy Decision Tree
  */
private[tree] object FuzzyDecisionTreeMetadata extends Logging {

  /**
    * The method builds the metadata for the problem starting from the input dataset and
    * the strategy used to learn the Fuzzy Decision Tree
    *
    * @param input the distributed dataset
    * @param strategy the strategy used to learn the Fuzzy Decision Tree
    * @return metadata of the problem
    */
  private def buildFDTMetadata(
      input: RDD[LabeledPoint],
      strategy: FDTStrategy): FuzzyDecisionTreeMetadata = {

    // Set number of features
    val numFeatures = input.map(_.features.size).take(1).headOption.getOrElse {
      throw new IllegalArgumentException(s"Fuzzy Decision Tree requires size of input RDD > 0, " +
        s"but was given by empty one.")
    }

    // Set number of examples and the number of possible bins
    val numExamples = input.count()
    val maxPossibleBins = math.min(strategy.maxBins, numExamples).toInt
    
    // We check the number of bins here against maxPossibleBins.
    // This needs to be checked here instead of in Strategy since maxPossibleBins can be modified
    // based on the number of training examples (in case of subsampling).
    if (strategy.categoricalFeaturesInfo.nonEmpty) {
      val maxCategoriesPerFeature = strategy.categoricalFeaturesInfo.values.max
      val maxCategory =
        strategy.categoricalFeaturesInfo.find(_._2 == maxCategoriesPerFeature).get._1
      require(maxCategoriesPerFeature <= maxPossibleBins,
        s"Fuzzy Decision Tree requires maxBins (= $maxPossibleBins) to be at least as large as the " +
        s"number of values in each feature, but categorical feature $maxCategory " +
        s"has $maxCategoriesPerFeature values. Considering remove this and other " +
        "features with a large number of values, or add more training examples.")
    }
    
    // For each feature create Fuzzy Sets based on thresholds
    val fuzzySetFeaturesArity = mutable.Map.empty[Int, Array[FuzzySet]]
    strategy.thresholdsFeatureInfo.foreach{ case (index, thresholds) =>
        fuzzySetFeaturesArity(index) = TriangularFuzzySet.createFuzzySets(thresholds)
    }
    
    // Calculate arity for each feature
    val numBins = {
      val unorderedMap = strategy.categoricalFeaturesInfo.map{ case (featureIndex, arity) =>
        (featureIndex, arity)
      } ++ fuzzySetFeaturesArity.map{ case (featureIndex, fuzzySets) =>
        (featureIndex, fuzzySets.length) 
      }
      SortedMap[Int, Int](unorderedMap.toSeq: _*)
    }
        
    // Set number of bins for each feature and feature that should be treated as unordered
    val (numFeatureBin: Map[Int, Int], unorderedFeatures: Set[Int]) = {
      SplitType.fromString(strategy.splitType) match {
        case Binary =>   
          /*
           * We decide if a feature should be treated as unordered feature and we set 
           * the right number of bins for each feature.
           * 
           * First, we check the number of bins here against maxPossibleBins.
           * This needs to be checked here instead of in Strategy since maxPossibleBins can be modified
           * based on the number of training examples.
           */
          val maxCategoriesPerFeature = numBins.values.max 
          val maxCategory = numBins.find(_._2 == maxCategoriesPerFeature).get._1
          require(maxCategoriesPerFeature <= maxPossibleBins,
            s"Fuzzy Decision Tree requires maxBins (= $maxPossibleBins) to be at least as large as the " +
            s"number of values in feature, when split is set to binary, but feature $maxCategory " +
            s"has $maxCategoriesPerFeature values. Considering remove this and other " +
            s"features with a large number of values, or add more training examples.\n" +
            s"For values, we refer to fuzzy sets and categories in case of " + 
             "continuous and categorical features respectively")
          
          val unorderedFeatures = new mutable.HashSet[Int]()
          val numFeatureBin = Array.fill[Int](numFeatures)(maxPossibleBins)
          if (strategy.numClasses > 2) {
            // Multi-class classification
            val maxCategoriesForUnorderedFeature =
                ((math.log(maxPossibleBins / 2 + 1) / math.log(2.0)) + 1).floor.toInt
            numBins.foreach { case (featureIndex, arity) =>
              if (arity > 1) {
                // Decide if some features should be treated as unordered features,
                //  which require 2 * ((1 << numCategories - 1) - 1) bins.
                // We do this check with log values to prevent overflows in case numCategories is large.
                // The next check is equivalent to: 2 * ((1 << numCategories - 1) - 1) <= maxBins
                if (arity <= maxCategoriesForUnorderedFeature && !fuzzySetFeaturesArity.contains(featureIndex)) {
                  unorderedFeatures.add(featureIndex)
                  numFeatureBin(featureIndex) = numUnorderedBins(arity)
                } else {
                  numFeatureBin(featureIndex) = arity
                }
              }else{
                throw new IllegalArgumentException(s"Fuzzy Decision Tree requires that all features have more " +
                     s"than one value, but only $arity value has been inserted for feature $featureIndex. " +
                     s"Please considered to remove this feature before applying for FDT.")
              }
            }
          } else {
            // Binary classification
            numBins.foreach { case (featureIndex, arity) =>
              if (arity > 1) {
                numFeatureBin(featureIndex) = arity
              }else{
                 throw new IllegalArgumentException(s"Fuzzy Decision Tree requires that all features have more " +
                     s"than one value, but only $arity value has been inserted for feature $featureIndex. " +
                     s"Please considered to remove this feature before applying for FDT.") 
              }
            }
          }
          (numFeatureBin.zipWithIndex.map(_.swap).toMap, unorderedFeatures.toSet)
        case Multi =>
          /*
           * In this case we just emit the arity of each feature and an empty unordered features list
           */
           (numBins, Set.empty[Int])
        case _ => throw new IllegalArgumentException(s"Fuzzy Decision Tree requires that split types are " +
                      s"binary or multi branches")
      }
    
    }  
    // Set a map (featureId, fuzzySetIndexInPartition) -> fuzzyId
    val featureOffset = numBins.scanLeft(0)((total, nBins) => total +  nBins._2).toArray    
    val featureIndexAndIndexInPartition = Array.fill[(Int, Int)](featureOffset.last)((0,0))
    strategy.categoricalFeaturesInfo.foreach{ case (featureId, arity) =>
      val offset = featureOffset(featureId)
      (0 until arity).foreach { indexPartition =>
        featureIndexAndIndexInPartition(offset+indexPartition) = (featureId, indexPartition)      
      }
    }   
    fuzzySetFeaturesArity.foreach{ case (featureId, fuzzySets) =>
      val offset = featureOffset(featureId)
      fuzzySets.indices.foreach { indexPartition =>
        featureIndexAndIndexInPartition(offset+indexPartition) = (featureId, indexPartition)      
      }
    }
    val featureIndexAndIndexInPartitionToId = featureIndexAndIndexInPartition.zipWithIndex.toMap
    
    // Set a map fuzzySetId -> (featureId, fuzzySetIndexInPartition)
    val idToFeatureIndexAndIndexInPartition = featureIndexAndIndexInPartitionToId.map(_.swap)

    // Set impurity and t-norm
    val impurity = FuzzyImpurities.fromString(strategy.impurity)
    val tNorm = TNorms.fromString(strategy.tNorm)

    //Create and return the metadata instance of the problem
    new FuzzyDecisionTreeMetadata(numFeatures, numExamples, strategy.numClasses, 
      strategy.categoricalFeaturesInfo, fuzzySetFeaturesArity.toMap, numFeatureBin,
      unorderedFeatures, featureIndexAndIndexInPartitionToId, idToFeatureIndexAndIndexInPartition,
      impurity, tNorm, strategy.maxDepth, numFeatureBin.values.max, strategy.minInstancesPerNode,
      strategy.minFuzzyInstancesPerNode, strategy.minImpurityRatioPerNode, strategy.minInfoGain)
  }
  
  /**
   * The method builds the metadata for the problem starting from the input dataset and
   * the strategy used to learn the Fuzzy Decision Tree
   *
   * @param input the distributed dataset
   * @param strategy the strategy used to learn the Fuzzy Decision Tree
   * @return metadata of the problem.
   */
  def buildMetadata(
      input: RDD[LabeledPoint],
      strategy: FDTStrategy): FuzzyDecisionTreeMetadata = {
    buildFDTMetadata(input, strategy)
  }
  
   /**
   * Given the arity of a feature (arity = number of values (both categories or fuzzy sets)),
   * return the number of bins for the feature if it is to be treated as an unordered feature.
   * There is 1 split for every partitioning of categories into 2 disjoint, non-empty sets.
   * There are math.pow(2, arity - 1) - 1 such splits.
   * Each split has 2 corresponding bins (used only in case of Fuzzy Binary Decision Tree).
   */
  private def numUnorderedBins(arity: Int): Int = 2 * ((1 << arity - 1) - 1)

}
