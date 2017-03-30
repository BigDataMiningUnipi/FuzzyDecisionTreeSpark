package iet.unipi.bigdatamining.classification.tree.model

/**
  * Predicted value for a node
  *
  * @param uLabels array that stores for each label the sum of membership degrees
  *                of all points in the dataset from the root to the node
  * @param frequencyLabels array that stores for each label the count
  *                of all points in the dataset from the root to the node
  * @since 1.0
  */
class PredictStats(
                    val uLabels: Array[Double],
                    val frequencyLabels: Array[Double]) extends Serializable {

  /**
    * Return a string with a summary of the prediction stats.
    */
  override def toString: String = "[uL " + uLabels.mkString("[",";","]") +
    s"] - [fL " + frequencyLabels.mkString("[",";","]") + "]"

  override def equals(other: Any): Boolean = {
    other match {
      case p: PredictStats => uLabels.sameElements(p.uLabels) && frequencyLabels.sameElements(p.frequencyLabels)
      case _ => false
    }
  }

  override def hashCode: Int = {
    val javaULabel: java.util.Collection[java.lang.Double] =
      java.util.Arrays.asList(uLabels.map(java.lang.Double.valueOf): _*)
    val javaFrequencyLabel: java.util.Collection[java.lang.Double] =
      java.util.Arrays.asList(frequencyLabels.map(java.lang.Double.valueOf): _*)
    com.google.common.base.Objects.hashCode(javaULabel, javaFrequencyLabel)
  }

  /**
    * Return the number of instances of the node
    */
  def totalFreq: Double = frequencyLabels.sum

  /**
    * Return the sum of association degree of all points that belong to the node
    */
  def totalU: Double = uLabels.sum

  /**
    * Return true if the frequency is greater than 0 in only one labels
    */
  def isMonoLabel: Boolean = {
    frequencyLabels.count(_ > 0) == 1
  }

  /**
    * Return the ratio between uLabel and of the most frequent
    * label according association degree and total frequency
    */
  def getMaxLabelPredictionRatio: Double = {
    val maxLabel = uLabels.indices.reduceLeft{(x,y) =>
      if (uLabels(x) >= uLabels(y)) x else y
    }
    uLabels(maxLabel) / totalU
  }

  /**
    * Return the weighted frequency of each class label.
    * The i-th element is computed as freq(i)/totalFreq
    */
  def getWeightedFreq: Array[Double] = {
    if (totalFreq > 0) frequencyLabels.map(_/totalFreq) else Array.fill[Double](frequencyLabels.length)(0D)
  }

  /**
    * Return the weighted membership degree of each class label.
    * The i-th element is computed as uLabels(i)/totalFreq
    */
  def getWeightedMD: Array[Double] = {
    if (totalU > 0) uLabels.map(_/totalU) else  Array.fill[Double](uLabels.length)(0D)
  }

}

/**
  * Companion object of PredictStats.
  * It implements the constructor for the companion class and  methods
  */
private[tree] object PredictStats {

  /**
    * Utility method to quickly create an empty [[PredictStats]] class
    * @return a new empty [[PredictStats]] class where both association degrees and frequencies are empty arrays
    */
  def empty: PredictStats = {
    apply(Array.empty[Double], Array.empty[Double])
  }

  /**
    * Apply function of the object
    *
    * @param uLabels array that stores for each label the sum of membership degrees
    *                of all points in the dataset from the root to the node
    * @param frequencyLabels array that stores for each label the count
    *                of all points in the dataset from the root to the node
    * @return a new instance of [[PredictStats]]
    */
  def apply(uLabels: Array[Double],
            frequencyLabels: Array[Double]): PredictStats = {
    new PredictStats(uLabels, frequencyLabels)
  }

}