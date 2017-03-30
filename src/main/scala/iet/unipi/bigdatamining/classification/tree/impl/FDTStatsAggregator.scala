package iet.unipi.bigdatamining.classification.tree.impl

import iet.unipi.bigdatamining.classification.tree.impurity.{FuzzyEntropy, FuzzyEntropyAggregator, FuzzyImpurityAggregator, FuzzyImpurityCalculator}


/**
 * Fuzzy Decision Tree statistics aggregator for a node (both binary or multi split).
 * This holds a flat array of statistics for a set of (features, bins)
 * and helps with indexing.
 * This class is abstract to support learning with and without feature subsampling.
 */
private[tree] class FDTStatsAggregator(
    val metadata: FuzzyDecisionTreeMetadata) extends Serializable {

  /**
   * [[iet.unipi.bigdatamining.classification.tree.impurity.FuzzyImpurityAggregator]] instance
    * specifying the impurity type.
   */
  val fuzzyImpurityAggregator: FuzzyImpurityAggregator = metadata.impurity match {
    case FuzzyEntropy => new FuzzyEntropyAggregator(metadata.numClasses)
    case _ => throw new IllegalArgumentException(s"Bad impurity parameter: ${metadata.impurity}")
  }
  
  /**
   * Number of elements used for the sufficient statistics of each bin.
   * It corresponds to the number of class labels
   */
  private val statsSize: Int = fuzzyImpurityAggregator.statsSize
  
  /**
   * Number of bins for each feature. This is indexed by the feature index.
   */
  private val numBins: Array[Int] = {
      Array.tabulate(metadata.numFeatureBin.size) { featureIndex =>
        metadata.numFeatureBin.getOrElse(featureIndex, 0)
      }
  }
  
  /**
   * Offset for each feature for calculating indices into the [[uStats]] and [[freqStats]] arrays.
   */
  private val featureOffsets: Array[Int] = {
    numBins.scanLeft(0)((total, nBins) => total + statsSize * nBins)
  }
  
  /**
   * Total number of elements stored in this aggregator
   */
  private val allStatsSize: Int = featureOffsets.last
  
  /**
   * Flat array of elements.
   * Index for start of stats for a (feature, bin) is:
   *   index = featureOffsets(featureIndex) + binIndex * statsSize
   * The array stores the sum of membership degrees for each (feature, bin)
   * and class. 
   */
  private val uStats: Array[Double] = new Array[Double](allStatsSize)

  /**
   * Flat array of elements.
   * As [[uStats]] property index for start of stats for a (feature, bin) is:
   *   index = featureOffsets(featureIndex) + binIndex * statsSize
   * The array stores the frequency of points for each (feature, bin)
   * and class. It is used to know how many instances belong to a specific
   * node and to check the minInstancesPerNode stop condition when a
   * new child node has to be created
   */
  private val freqStats: Array[Double] = new Array[Double](allStatsSize)

  /**
   * Get an [[iet.unipi.bigdatamining.classification.tree.impurity.FuzzyImpurityCalculator]] for a given (node, feature, bin).
    *
    * @param featureOffset  For ordered features, this is a pre-computed (node, feature) offset
   *                           from [[getFeatureOffset]].
   */
  def getImpurityCalculator(featureOffset: Int, binIndex: Int): FuzzyImpurityCalculator = {
    fuzzyImpurityAggregator.getCalculator(uStats, featureOffset + binIndex * statsSize)
  }

  /**
   * Get the frequency of a given (feature, bin).
   * The returned array as the same length of statsSize 
   */
  def getFreqStats(featureOffset: Int, binIndex: Int): Array[Double] = {
    val offset = featureOffset + binIndex * statsSize
    freqStats.slice(offset, offset + statsSize)
  }
  
  /**
   * Update all stats both uStats and freqStats for a given (feature, bin) 
   * using the given label.
   * 
   * For uStats the corresponding bin is updated as follow:
   *    tNorm(uAi(xi), uAs(xi))  + binValue
   * where binValue is the current value of the given (feature, bin) 
   * Note that uAi(xi) and uAs(xi) have been already calculated and their values are stored
   * in TreePoint data structure
   *  
   * For freqStats the corresponding bin is updated as follow: 
   *    binValue
   * where binValue is the current value of the given (feature, bin)
   */
  def updateStats(featureIndex: Int, binIndex: Int, label: Double, 
      uAi: Double, uAs: Double): Unit = {
    if (uAi != 0D){
      // Update uStats
      val value = metadata.tNorm.calculate(uAi, uAs)
      val offset = featureOffsets(featureIndex) + binIndex * statsSize
      fuzzyImpurityAggregator.update(uStats, offset, label, value)
      // Update freqStats
      freqStats(offset + label.toInt) += 1
    }
  }
  
  /**
   * Faster version of [[updateStats]].
   * Update the stats for a given (feature, bin), using the given label.
    *
    * @param featureOffset  For ordered features, this is a pre-computed feature offset
   *                           from [[getFeatureOffset]].
   */
  def featureUpdateStats(featureOffset: Int, binIndex: Int, label: Double,
      uAi: Double, uAs: Double): Unit = {
    if (uAi != 0D){
      // Update uStats
      val value = metadata.tNorm.calculate(uAi, uAs)
      fuzzyImpurityAggregator.update(uStats, featureOffset + binIndex * statsSize,
        label, value)
       // Update freqStats
      freqStats((featureOffset + binIndex * statsSize) + label.toInt) += 1
    }
  }
  
  /**
   * Pre-compute feature offset for use with [[featureUpdateStats]] and [[updateStats]] methods.
   * For ordered features only.
   */
  def getFeatureOffset(featureIndex: Int): Int = featureOffsets(featureIndex)
  
  /**
   * Pre-compute feature offset for use with [[featureUpdateStats]] and [[updateStats]] methods.
   * For unordered features only.
   */
  def getLeftRightFeatureOffsets(featureIndex: Int): (Int, Int) = {
    val baseOffset = featureOffsets(featureIndex)
    (baseOffset, baseOffset + (numBins(featureIndex) >> 1) * statsSize)
  }
  
  /**
   * For a given feature, merge the stats for two bins.
   *
   * @param featureOffset  this is a pre-computed feature offset from [[getFeatureOffset]].
   * @param binIndex  The other bin is merged into this bin.
   * @param otherBinIndex  This bin is not modified.
   */
  def mergeForFeature(featureOffset: Int, binIndex: Int, otherBinIndex: Int): Unit = {
    val offset = featureOffset + binIndex * statsSize
    val otherOffset = featureOffset + otherBinIndex * statsSize
    // Update uStats
    fuzzyImpurityAggregator.merge(uStats, offset, otherOffset)
    // Update freqStats
    var i = 0
    while (i < statsSize) {
      freqStats(offset + i) += freqStats(otherOffset + i)
      i += 1
    }
  }
  
  /**
   * Merge this aggregator with another, and returns this aggregator.
   * This method modifies this aggregator in-place.
   */
  def merge(other: FDTStatsAggregator): FDTStatsAggregator = {
    require(allStatsSize == other.allStatsSize,
      s"DTStatsAggregator.merge requires that both aggregators have the same length stats vectors."
        + s" This aggregator is of length $allStatsSize, but the other is ${other.allStatsSize}.")

    (0 until allStatsSize).foreach { index =>
      uStats(index) += other.uStats(index)
      freqStats(index) += other.freqStats(index)
    }

    this
  }

  /**
    * The string representing statistics aggregator.
    * For debugging only porpoise.
    */
  def toDebugString: String = {
    val sb = new StringBuilder()
    sb.append("uStats: " + uStats.mkString("[", ":", "]") + "\n")
    sb.append("fStats: " + freqStats.mkString("[", ":", "]") + "\n")
    sb.toString
  }
  
}