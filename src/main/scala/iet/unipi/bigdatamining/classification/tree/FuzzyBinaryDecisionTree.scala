package iet.unipi.bigdatamining.classification.tree

import iet.unipi.bigdatamining.classification.tree.configuration.FDTStrategy
import iet.unipi.bigdatamining.classification.tree.impl.{FuzzyDecisionTreeMetadata, TreePoint}
import iet.unipi.bigdatamining.classification.tree.model.{FuzzyDecisionTreeModel, PredictStats}
import iet.unipi.bigdatamining.classification.tree.model.binary.{BinaryInformationGainStats, BinaryNode, BinarySplit}
import iet.unipi.bigdatamining.classification.tree.impl.FDTStatsAggregator
import iet.unipi.bigdatamining.classification.tree.impurity.FuzzyImpurityCalculator

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.immutable.Map
import org.apache.spark.Logging
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import iet.unipi.bigdatamining.fuzzy.fuzzyset.SingletonFuzzySet
import iet.unipi.bigdatamining.fuzzy.tnorm.TNorm

/**
  * A Companion Class that implements a Fuzzy Binary Decision Tree learning algorithm for classification.
  * It supports both continuous and categorical features.
  *
  * @param fdtStrategy The configuration parameters for the Fuzzy Binary Decision Tree algorithm
  * @since 1.0
  */
class FuzzyBinaryDecisionTree (private val fdtStrategy: FDTStrategy) extends Serializable with Logging{

  fdtStrategy.assertValid()

  /**
    * Method to train a decision tree model over an RDD
    *
    * @param input distributed training dataset, i.e. an RDD of [[org.apache.spark.mllib.regression.LabeledPoint]]
    * @return a Fuzzy Decision Tree Model that can be used for prediction
    *
    * @since 1.0
    */
  def run(input: RDD[LabeledPoint]): FuzzyDecisionTreeModel = {
    log.info(s"Building metadata")

    // Builds metadata from input parameters
    val metadata = FuzzyDecisionTreeMetadata.buildMetadata(input, fdtStrategy)

    log.info(s"Pre-computing splitting and converting dataset")

    /*
     *  Differently from DecisionTree of Spark Mllib, we find splits but not the corresponding bins.
     *  We recall that for continuous features, a fuzzy partition must be already
     *  defined and we treat them as categorical features, where each fuzzy sets
     *  is a possible value of the feature.
     */
    val splits = FuzzyBinaryDecisionTree.findSplits(metadata)

    // Bin feature values (TreePoint representation).
    // Cache input RDD for speedup during multiple passes.
    val treeInput = TreePoint.convertToTreeRDD(input, metadata).cache()

    // Check depth of the decision tree
    val maxDepth = fdtStrategy.maxDepth
    require(maxDepth <= 30,
      s"Fuzzy Decision Tree currently only supports maxDepth <= 30, but was given maxDepth = $maxDepth.")

    // Max memory usage for aggregates (Calculate more precisely)
    val maxMemoryUsage = fdtStrategy.maxMemoryInMB * 1024L * 1024L
    val maxMemoryPerNode = metadata.numFeatureBin.values.sum * metadata.numClasses * 8L

    require(maxMemoryPerNode <= maxMemoryUsage,
      s"RandomForest/DecisionTree given maxMemoryInMB = ${fdtStrategy.maxMemoryInMB}," +
        s" which is too small for the given features." +
        s"  Minimum value = ${maxMemoryPerNode / (1024L * 1024L)}")

    log.info(s"Building Fuzzy Binary Decision Tree")

    /*
     * The main idea here is to perform group-wise training of the fuzzy decision tree nodes thus
     * reducing the passes over the data from (# nodes) to (# nodes / maxNumberOfNodesPerGroup).
     * Each data sample is handled by a particular node (or it reaches a leaf and is not used
     * in lower levels).
     */
    // FIFO queue of nodes to train: (treeIndex, node)
    val nodeQueue = new mutable.Queue[BinaryNode]()

    // Allocate and queue root nodes.
    val root = BinaryNode.emptyNode(nodeIndex = 1)
    nodeQueue.enqueue(root)
    while (nodeQueue.nonEmpty) {
      // Collect some nodes to split, and choose features for each node (if subsampling).
      // Each group of nodes may come from one or multiple levels.
      val nodesToSplit = FuzzyBinaryDecisionTree.selectNodesToSplit(nodeQueue, maxMemoryUsage, metadata)
      // Sanity check (should never occur):
      assert(nodesToSplit.length > 0,
        s"FuzzyBinaryDecisionTree selected empty nodesForGroup. Error for unknown reason.")

      log.info(s"Splitting ${nodesToSplit.length} nodes")

      // Choose node splits, and enqueue new nodes as needed.
      FuzzyBinaryDecisionTree.findBestSplits(treeInput, metadata, root,
        nodesToSplit, splits, nodeQueue)
    }

    treeInput.unpersist()

    log.info(s"Fuzzy Binary Decision Tree built")

    val idToFeatureIdFuzzySet = metadata.idToFeatureIndexAndIndexInPartition.map{ case (k,v) =>
      if (metadata.isContinuous(v._1))
        (k, (v._1, metadata.fuzzySetFeatures(v._1)(v._2)))
      else
        (k, (v._1, new SingletonFuzzySet(v._2)))
    }

    log.info(s"Returning the Fuzzy Binary Decision Tree model")

    // Create and return the model
    new FuzzyDecisionTreeModel(root, metadata.tNorm, idToFeatureIdFuzzySet)
  }

}

/**
  * Companion object of FuzzyBinaryDecisionTree
  */
object FuzzyBinaryDecisionTree {

  private final def SPLIT_TYPE: String = "binary_split"

  /**
    * Method to train a fuzzy decision tree.
    *
    * @param input distributed training dataset, i.e. an RDD of [[org.apache.spark.mllib.regression.LabeledPoint]]
    * @param impurity name of the fuzzy impurity
    * @param tNorm name of the t-norm
    * @param maxDepth maximum depth of Fuzzy Decision Tree
    * @param maxBins maximum number of bins for each feature
    * @param numClasses number of class labels. It can take values {0, ..., numClasses - 1}.
    * @param categoricalFeaturesInfo  a map that contains the categorical feature indexes as key
    *                                and the number of categorical values as value of the map
    * @param thresholdsFeatureInfo a map that contains the continuous feature indexes as key
    *                                and the an array of points as value of the map.
    *                                Points represent the threshold on which build the fuzzy partition
    * @param minInstancesPerNode minimum number of examples that each leaf must contain (default 1)
    * @param minImpurityRatioPerNode minimum ratio thresholds of impurity that must be true in each node
    *                                (default 0)
    * @param minInfoGain minimum information gain threshold that must be true in each node (default 0).
    * @param subsamplingRate ratio of subsampling (default 1, i.e. all dataset is considered)
    * @return Fuzzy Decision Tree Model that can be used for classification
    * @since 1.0
    */
  def train(
             input: RDD[LabeledPoint],
             impurity: String = "fuzzy_entropy",
             tNorm: String = "Product",
             maxDepth: Int = 10,
             maxBins: Int = 32,
             numClasses: Int = 2,
             categoricalFeaturesInfo: Map[Int, Int] = Map.empty[Int, Int],
             thresholdsFeatureInfo: Map[Int, Array[Double]] = Map.empty[Int, Array[Double]],
             minInstancesPerNode: Int = 1,
             minFuzzyInstancesPerNode: Double = 0D,
             minImpurityRatioPerNode: Double = 1D,
             minInfoGain: Double = 0.000001,
             subsamplingRate: Double = 1D): FuzzyDecisionTreeModel = {
    val strategy = new FDTStrategy(SPLIT_TYPE, impurity, tNorm, maxDepth, maxBins, numClasses,
      categoricalFeaturesInfo, thresholdsFeatureInfo, minInstancesPerNode, minFuzzyInstancesPerNode,
      minImpurityRatioPerNode, minInfoGain, subsamplingRate)
    new FuzzyBinaryDecisionTree(strategy).run(input)
  }

  /**
    * Java-friendly API for [[iet.unipi.bigdatamining.classification.tree.FuzzyBinaryDecisionTree#train]]
    */
  def trainFromJava(
                     input: JavaRDD[LabeledPoint],
                     impurity: java.lang.String = "fuzzy_entropy",
                     tNorm: java.lang.String = "Product",
                     maxDepth: java.lang.Integer = 10,
                     maxBins: java.lang.Integer = 32,
                     numClasses: java.lang.Integer = 2,
                     categoricalFeaturesInfo: java.util.Map[java.lang.Integer, java.lang.Integer]
                        = new java.util.HashMap[java.lang.Integer, java.lang.Integer](),
                     thresholdsFeatureInfo: java.util.Map[java.lang.Integer, java.util.List[java.lang.Double]]
                        = new java.util.HashMap[java.lang.Integer, java.util.List[java.lang.Double]](),
                     minInstancesPerNode: java.lang.Integer = 1,
                     minFuzzyInstancesPerNode: java.lang.Double = 0D,
                     minImpurityRatioPerNode: java.lang.Double = 1D,
                     minInfoGain: java.lang.Double = 0.000001,
                     subsamplingRate: java.lang.Double = 1D): FuzzyDecisionTreeModel = {
    val scalaThresholdsFeatureInfo = mutable.Map.empty[Int, Array[Double]]
    thresholdsFeatureInfo.asScala.foreach{ case (key, values) =>
      val cutPoints = Array.fill[Double](values.size())(0D)
      Range(0, values.size()).foreach{ i =>
        cutPoints(i) = values.get(i)
      }
      scalaThresholdsFeatureInfo(key.intValue()) = cutPoints
    }
    train(input.rdd, impurity, tNorm, maxDepth.intValue(), maxBins.intValue(), numClasses.intValue(),
      categoricalFeaturesInfo.asInstanceOf[java.util.Map[Int, Int]].asScala.toMap,
      scalaThresholdsFeatureInfo.toMap, minInstancesPerNode.intValue(),
      minFuzzyInstancesPerNode.doubleValue, minImpurityRatioPerNode.doubleValue(),
      minInfoGain.doubleValue(), subsamplingRate.doubleValue())
  }

  /**
    * Get the node indexes and the activation degree corresponding to this data point.
    * This function mimics prediction, passing an example from the root node down to a leaf
    * or no-split node; that indexes of each fired node with the associated
    * activation degree are returned.
    *
    * @param node  Node in tree from which to classify the given data point.
    * @param binnedFeatures  Binned feature vector for data point.
    * @param tNorm for computing the activation degree
    * @param currAD current activation degree
    * @param mdHistory array of the featureId considered until this node
    * @return  Leaf index if the data point reaches a leaf.
    *          Otherwise, last node reachable in tree matching this example.
    *          Every tuple has the activation degree from the root to the node
    *          identified by the returned ids. The activation degree is computed
    *          according the given tNorm
    *          Note: This is the global node index, i.e., the index used in the tree.
    *                This index is different from the index used during training a particular
    *                group of nodes on one call to [[findBestSplits()]].
    */
  private def getFiredNodes(
                             node: BinaryNode,
                             binnedFeatures: Array[Array[(Int, Double)]],
                             tNorm: TNorm,
                             currAD: Double = 1,
                             mdHistory: Map[Int, Int] = Map.empty[Int, Int]): Iterable[(Int,Double)] = {
    if (node.isLeaf || node.split.isEmpty){
      // Node is either leaf, or has not yet been split.
      List[(Int,Double)]((node.id, currAD))
    } else {
      // Continuous or Categorical features are processed in the same way
      var childrenIds = Iterable.empty[(Int, Double)]
      val featureIndex = node.split.get.feature
      val firedChildrenIds = binnedFeatures(featureIndex).toMap
      if (firedChildrenIds.keySet.subsetOf(node.split.get.categoryIndexes.toSet)){
        // All fired values are in the left child
        // Note that we don't calculate tNorm since the sum of all membership degree
        // is equal to 1
        childrenIds = childrenIds ++ getFiredNodes(node.leftChild.get,
          binnedFeatures, tNorm, currAD, mdHistory)
      }else{
        val leftIndex = node.split.get.categoryIndexes.toSet.intersect(firedChildrenIds.keySet)
        if (leftIndex.isEmpty){
          // All fired values are in the right child
          // Note that we don't calculate tNorm since the sum of all membership degree
          // is equal to 1
          childrenIds = childrenIds ++ getFiredNodes(node.rightChild.get,
            binnedFeatures, tNorm, currAD, mdHistory)
        }else{
          // One value belongs to left child and one to right child
          val rightIndex = firedChildrenIds.keySet.diff(leftIndex)
          var leftCurrAD = currAD
          var rightCurrAD = currAD
          if (mdHistory.get(node.split.get.feature).isEmpty){
            leftCurrAD = tNorm.calculate(currAD, firedChildrenIds(leftIndex.head))
            rightCurrAD = tNorm.calculate(currAD, firedChildrenIds(rightIndex.head))
          }
          childrenIds = childrenIds ++
            getFiredNodes(node.leftChild.get,
              binnedFeatures, tNorm, leftCurrAD, mdHistory + (node.split.get.feature -> leftIndex.head)) ++
            getFiredNodes(node.rightChild.get,
              binnedFeatures, tNorm, rightCurrAD, mdHistory + (node.split.get.feature -> rightIndex.head))
        }
      }
      childrenIds
    }
  }

  /**
    * Helper for binSeqOp for updating the sufficient statistics for each feature.
    * It is called when there are features both unordered or ordered
    *
    * @param agg  Array storing aggregate calculation, with a set of sufficient statistics for
    *             each (feature, bin).
    * @param treePoint  Data point being aggregated.
    * @param uAs membership degree of the point
    * @param splits array of BinarySplit
    * @param unorderedFeatures a set of the indexes of unordered features
    * @param nodeSplitHistory a map that contains the index of feature as key
    *                         and a set of indexes of fuzzy sets as value
    */
  private def mixedBinSeqOpAggregatorHelper(
                                             agg: FDTStatsAggregator,
                                             treePoint: TreePoint,
                                             uAs: Double,
                                             splits: Array[Array[BinarySplit]],
                                             unorderedFeatures: Set[Int],
                                             nodeSplitHistory: Map[Int, Set[Int]]) = {
    val label = treePoint.label
    val numFeatures = agg.metadata.numFeatures

    // Iterate over features.
    (0 until numFeatures).foreach{ featureIndex =>
      if (unorderedFeatures.contains(featureIndex)) {
        // Unordered feature
        val (leftNodeFeatureOffset, rightNodeFeatureOffset) =
          agg.getLeftRightFeatureOffsets(featureIndex)

        // Update the left or right bin for each split.
        val numSplits = agg.metadata.numSplits(featureIndex)
        val binIndexes = treePoint.binnedFeatures(featureIndex)
        binIndexes.foreach{ binIndex =>
          val uAi = if (nodeSplitHistory.contains(featureIndex)) 1D else binIndex._2
          if (!nodeSplitHistory.contains(featureIndex) || nodeSplitHistory(featureIndex).contains(binIndex._1)){
            (0 until numSplits).foreach{splitIndex =>
              if (splits(featureIndex)(splitIndex).categoryIndexes.contains(binIndex._1)) {
                agg.featureUpdateStats(leftNodeFeatureOffset, splitIndex, label, uAi, uAs)
              }else{
                agg.featureUpdateStats(rightNodeFeatureOffset, splitIndex, label, uAi, uAs)
              }
            }
          }
        }
      }else{
        // Ordered feature
        val binIndexes = treePoint.binnedFeatures(featureIndex)
        binIndexes.foreach{ binIndex =>
          val uAi = if (nodeSplitHistory.contains(featureIndex)) 1D else binIndex._2
          if (!nodeSplitHistory.contains(featureIndex) || nodeSplitHistory(featureIndex).contains(binIndex._1)){
            agg.updateStats(featureIndex, binIndex._1, label, uAi, uAs)
          }
        }
      }
    }

  }

  /**
    * Helper for binSeqOp for updating the sufficient statistics for each feature.
    * It is called only if all features are unordered.
    *
    * @param agg  Array storing aggregate calculation, with a set of sufficient statistics for
    *             each (feature, bin).
    * @param treePoint  Data point being aggregated.
    * @param uAs membership degree of the point
    * @param nodeSplitHistory a map that contains the index of feature as key
    *                         and a set of indexes of fuzzy sets as value
    */
  private def orderedBinSeqOpAggregatorHelper(
                                               agg: FDTStatsAggregator,
                                               treePoint: TreePoint,
                                               uAs: Double,
                                               nodeSplitHistory: Map[Int, Set[Int]]) = {
    val label = treePoint.label
    val numFeatures = agg.metadata.numFeatures
    (0 until numFeatures).foreach{featureIndex =>
      val binIndexes = treePoint.binnedFeatures(featureIndex)
      binIndexes.foreach{ binIndex =>
        val uAi = if (nodeSplitHistory.contains(featureIndex)) 1D else binIndex._2
        if (!nodeSplitHistory.contains(featureIndex) || nodeSplitHistory(featureIndex).contains(binIndex._1))
          agg.updateStats(featureIndex, binIndex._1, label, uAi, uAs)

      }
    }
  }

  /**
    * Given a group of nodes, this method finds the best split for each node.
    *
    * @param input Training data: RDD of [[iet.unipi.bigdatamining.classification.tree.impl.TreePoint]]
    * @param metadata Learning and dataset metadata
    * @param root Root node of the tree.
    * @param nodesToSplit array of nodes to be split
    * @param splits possible splits for all features, indexed (numFeatures)(numSplits)
    * @param nodeQueue  Queue of nodes to split, with values (treeIndex, node).
    *                   Updated with new non-leaf nodes which are created.
    */
  private[tree] def findBestSplits(
                                    input: RDD[TreePoint],
                                    metadata: FuzzyDecisionTreeMetadata,
                                    root: BinaryNode,
                                    nodesToSplit: Array[BinaryNode],
                                    splits: Array[Array[BinarySplit]],
                                    nodeQueue: mutable.Queue[BinaryNode]) = {
    /*
     * The high-level descriptions of the best split optimizations are noted here.
     *
     * * Group-wise training *
     * We perform bin calculations for groups of nodes to reduce the number of
     * passes over the data. Each iteration requires more computation and storage,
     * but saves several iterations over the data.
     *
     * * Bin-wise computation *
     * We first categorize each feature into a bin.
     * We exploit this structure to calculate aggregates for bins and then use these aggregates
     * to calculate information gain for each split. For continuous features, each bin is a fuzzy set.
     * For categorical feature each bin is the feature value.
     *
     * * Aggregation over partitions *
     * Instead of performing a flatMap/reduceByKey operation, we exploit the fact that we know
     * the number of splits in advance. Thus, we store the aggregates (at the appropriate
     * indices) in a single array for all bins and rely upon the RDD aggregate method to
     * drastically reduce the communication overhead.
     */

    // numNodes:  Number of nodes in this group
    val numNodes = nodesToSplit.length
    // map from node id to node index in the group
    val nodeIdToNodeIndexInGroup = (0 until numNodes).map(index => (nodesToSplit(index).id, index)).toMap

    /**
      * Performs a sequential aggregation over a partition for a particular node.
      *
      * For each feature, the aggregate sufficient statistics are updated for the relevant
      * bins.
      *
      * @param nodeIndexInGroup the index of the node in the group
      * @param agg Array storing aggregate calculation, with a set of sufficient statistics
      *            for each (node, feature, bin).
      * @param point Data point being aggregated.
      * @param uAs the activation degree from the root to the node  identified from nodeIndexInGroup
      */
    def nodeBinSeqOp(
                      nodeIndexInGroup: Int,
                      agg: Array[FDTStatsAggregator],
                      point: TreePoint,
                      uAs: Double) = {
      if (nodeIndexInGroup >= 0) {
        if (metadata.unorderedFeatures.isEmpty) {
          orderedBinSeqOpAggregatorHelper(agg(nodeIndexInGroup), point,
            uAs, nodesToSplit(nodeIndexInGroup).splitsHistory)
        }else{
          mixedBinSeqOpAggregatorHelper(agg(nodeIndexInGroup), point,
            uAs, splits, metadata.unorderedFeatures,
            nodesToSplit(nodeIndexInGroup).splitsHistory)
        }
      }
    }

    /**
      * Performs a sequential aggregation over a partition.
      *
      * Each data point contributes to one node. For each feature,
      * the aggregate sufficient statistics are updated for the relevant bins.
      *
      * @param agg  Array storing aggregate calculation, with a set of sufficient statistics for
      *             each (node, feature, bin).
      * @param point Data point being aggregated.
      * @return an array of statistic aggregators
      */
    def binSeqOp(
                  agg: Array[FDTStatsAggregator],
                  point: TreePoint): Array[FDTStatsAggregator] = {
      val firedNodes = getFiredNodes(root, point.binnedFeatures, agg(0).metadata.tNorm)
      firedNodes.foreach { firedNode =>
        nodeBinSeqOp(nodeIdToNodeIndexInGroup.getOrElse(firedNode._1, -1), agg, point, firedNode._2)
      }

      agg
    }

    // In each partition, iterate all instances and compute aggregate stats for each node,
    // yield an (nodeIndex, nodeAggregateStats) pair for each node.
    // After a 'reduceByKey' operation,
    // stats of a node will be shuffled to a particular partition and be combined together,
    // then best splits for nodes are found there.
    // Finally, only best Splits for nodes are collected to driver to construct decision tree.
    val partitionAggregates =
      input.mapPartitions { points =>
        // Construct a nodeStatsAggregators array to hold node aggregate stats,
        // each node will have a nodeStatsAggregator
        // Note that now nodeIndex assumes the same values of nodeIndexInGroup
        val nodeStatsAggregators = Array.tabulate(numNodes) { nodeIndex =>
          new FDTStatsAggregator(metadata)
        }

        // iterator all instances in current partition and update aggregate stats
        points.foreach(binSeqOp(nodeStatsAggregators, _))

        // transform nodeStatsAggregators array to (nodeIndex, nodeAggregateStats) pairs,
        // which can be combined with other partition using `reduceByKey`
        nodeStatsAggregators.view.zipWithIndex.map(_.swap).iterator
      }

    val nodeToBestSplits = partitionAggregates.reduceByKey((a, b) => a.merge(b))
      .map{ case (nodeIndexInGroup, aggStats) =>
        // Find best split for each node
        val (split: BinarySplit, stats: BinaryInformationGainStats, predict: PredictStats) =
          binsToBestSplit(aggStats, splits, nodesToSplit(nodeIndexInGroup))
        (nodeIndexInGroup, (split, stats, predict))
      }.collectAsMap()

    nodesToSplit.foreach{ node =>
      val nodeId = node.id
      val nodeIndexInGroup = nodeIdToNodeIndexInGroup(nodeId)
      val (split: BinarySplit, gainStats: BinaryInformationGainStats, predictStats: PredictStats) =
        nodeToBestSplits(nodeIndexInGroup)

      // Extract info for this node. Create children if not leaf.
      val isLeaf = ((gainStats.gain <= 0D) || (BinaryNode.indexToLevel(nodeId) == metadata.maxDepth)
        || (predictStats.getMaxLabelPredictionRatio >= metadata.minImpurityRatioPerNode))
      node.predictStats = predictStats
      node.isLeaf = isLeaf
      node.gainStats = Some(gainStats)
      node.impurity = gainStats.impurity

      if (!isLeaf){
        node.split = Some(split)

        val childIsLeaf = (BinaryNode.indexToLevel(nodeId) + 1) == metadata.maxDepth
        val leftChildIsLeaf = (childIsLeaf || (gainStats.leftImpurity <= 0D)
          || (gainStats.leftPredict.totalFreq < metadata.minInstancesPerNode)
          || (gainStats.leftPredict.totalU <= metadata.minFuzzyInstancesPerNode))

        val rightChildIsLeaf = (childIsLeaf || (gainStats.rightImpurity <= 0D)
          || (gainStats.rightPredict.totalFreq < metadata.minInstancesPerNode)
          || (gainStats.rightPredict.totalU <= metadata.minFuzzyInstancesPerNode))
        node.leftChild = Some(BinaryNode.apply(BinaryNode.leftChildIndex(nodeId),
          node.splitsHistory ++ Map[Int, Set[Int]]((split.feature, split.categoryIndexes.toSet)),
          gainStats.leftImpurity, leftChildIsLeaf,  gainStats.leftPredict))
        node.rightChild = Some(BinaryNode.apply(BinaryNode.rightChildIndex(nodeId),
          node.splitsHistory ++ Map[Int, Set[Int]]((split.feature,
            node.splitsHistory.getOrElse(split.feature, (0 until metadata.numCategories(split.feature)).toSet)
              .diff(split.categoryIndexes.toSet))),
          gainStats.rightImpurity, rightChildIsLeaf,  gainStats.rightPredict))

        // Enqueue left child and right child if they are not leaves
        if (!leftChildIsLeaf)
          nodeQueue.enqueue(node.leftChild.get)

        if (!rightChildIsLeaf)
          nodeQueue.enqueue(node.rightChild.get)

      }

    }

  }

  /**
    * Calculate the information gain for a given (feature, split) based upon left/right aggregates.
    *
    * @param leftFuzzyImpurityCalculator fuzzy impurity calculator for left child of the split
    * @param rightFuzzyImpurityCalculator fuzzy impurity calculator for right child of the split
    * @param metadata of the fuzzy decision tree to be trained
    * @param impurity of the node to be split
    * @return information gain and statistics for split
    */
  private def calculateGainForSplit(
                                     leftFuzzyImpurityCalculator: (FuzzyImpurityCalculator, Array[Double]),
                                     rightFuzzyImpurityCalculator:(FuzzyImpurityCalculator, Array[Double]),
                                     metadata: FuzzyDecisionTreeMetadata,
                                     impurity: Double): BinaryInformationGainStats = {
    /*
     *  Calculate Weighted Fuzzy Impurity WFent(Split;S)
     *  Note that since the partitions is strong, totalCount
     *  is equal to the frequency of the examples and represent
     *  the cardinality of D
     */
    val leftCardinality = leftFuzzyImpurityCalculator._1.count
    val rightCardinality = rightFuzzyImpurityCalculator._1.count
    val totalCount = leftCardinality + rightCardinality

    val leftFuzzyImpurities = leftFuzzyImpurityCalculator._1.calculate() // Note: This equals 0 if count = 0
    val rightFuzzyImpurities = rightFuzzyImpurityCalculator._1.calculate()

    val wFuzzyImpurity = (leftFuzzyImpurities * leftCardinality
      + rightFuzzyImpurities * rightCardinality) / totalCount

    // Calculate gain (impurity - WFent)
    val gain = impurity - wFuzzyImpurity

    // If information gain doesn't satisfy minimum information gain,
    // then this split is invalid and return invalid information gain stats.
    if (gain < metadata.minInfoGain) {
      return BinaryInformationGainStats.invalidInformationGainStats
    }

    // calculate left and right predict
    val leftPredict = calculatePredict(leftFuzzyImpurityCalculator._1.stats, leftFuzzyImpurityCalculator._2)
    val rightPredict = calculatePredict(rightFuzzyImpurityCalculator._1.stats, rightFuzzyImpurityCalculator._2)

    new BinaryInformationGainStats(gain, impurity,
      leftFuzzyImpurities, rightFuzzyImpurities,
      leftPredict, rightPredict)
  }

  /**
    * Given a membership degrees and frequencies stats
    *
    * @param uStats array that stores for each label the sum of membership degrees
    *                of all points in the dataset from the root to the node
    * @param freqStats array that stores for each label the count
    *                of all points in the dataset from the root to the node
    * @return a new instance predictStats
    */
  private def calculatePredict(
                                uStats: Array[Double],
                                freqStats: Array[Double]): PredictStats = {
    PredictStats(uStats, freqStats)
  }

  /**
    * Calculate predict value for current node, given stats of the two splits.
    * Note that this function is called only once for each tree and
    * it works only if fuzzy partition is strong.
    *
    * @param leftChildFuzzyImpurityCalculator fuzzy impurity calculator for left child node
    * @param rightChildFuzzyImpurityCalculator fuzzy impurity calculator for right child node
    * @return predict value and impurity for current node
    */
  private def calculatePredictImpurity(
      leftChildFuzzyImpurityCalculator: (FuzzyImpurityCalculator, Array[Double]),
      rightChildFuzzyImpurityCalculator: (FuzzyImpurityCalculator, Array[Double])): (PredictStats, Double) = {

    // Calculate Impurity
    val parentFuzzyImpurityCalculator = leftChildFuzzyImpurityCalculator._1.copy
    parentFuzzyImpurityCalculator.add(rightChildFuzzyImpurityCalculator._1)
    val parentFreqStats = (leftChildFuzzyImpurityCalculator._2, rightChildFuzzyImpurityCalculator._2)
      .zipped
      .map(_+_)
    val impurity = parentFuzzyImpurityCalculator.calculate()
    // Calculate PredictStats
    val predict = calculatePredict(parentFuzzyImpurityCalculator.stats, parentFreqStats)
    (predict, impurity)
  }

  /**
    * Find the best split for a node.
    *
    * @param binAggregates Bin statistics.
    * @param splits array of BinarySplit
    * @param node on which calculate the best splits
    * @return tuple for best split: (Split, information gain, prediction at node)
    */
  private def binsToBestSplit(
                               binAggregates: FDTStatsAggregator,
                               splits: Array[Array[BinarySplit]],
                               node: BinaryNode): (BinarySplit, BinaryInformationGainStats, PredictStats) = {
    // Calculate predict and impurity if current node is top node
    var predictWithImpurity: Option[(PredictStats, Double)] = if (node.isRoot) {
      None
    } else {
      Some((node.predictStats, node.impurity))
    }
    // For each feature calculate the gain and select the best (feature, split)
    val (bestSplit, bestSplitStats) = {
      (0 until binAggregates.metadata.numFeatures).map { featureIndex =>
        val numSplits = binAggregates.metadata.numSplits(featureIndex)
        val nodeFeatureSplitsHistory = node.splitsHistory.getOrElse(featureIndex, null)
        if (binAggregates.metadata.isContinuous(featureIndex)){
          // Cumulative sum (scanLeft) of bin statistics.
          // Afterwards, binAggregates for a bin is the sum of aggregates for
          // that bin + all preceding bins.
          val nodeFeatureOffset = binAggregates.getFeatureOffset(featureIndex)
          (0 until numSplits).foreach{splitIndex =>
            binAggregates.mergeForFeature(nodeFeatureOffset, splitIndex + 1, splitIndex)
          }
          // Find best split.
          val (minFuzzySet, maxFuzzySet) = if (nodeFeatureSplitsHistory == null)
            (0, numSplits)
          else
            (nodeFeatureSplitsHistory.min, nodeFeatureSplitsHistory.max)
          val (bestFeatureSplitIndex, bestFeatureGainStats) =
            if (minFuzzySet == maxFuzzySet)
              (minFuzzySet, BinaryInformationGainStats.invalidInformationGainStats)
            else
              (minFuzzySet until maxFuzzySet).map { case splitIdx =>
                val leftChildStats: (FuzzyImpurityCalculator, Array[Double]) = (binAggregates.getImpurityCalculator(nodeFeatureOffset, splitIdx),
                  binAggregates.getFreqStats(nodeFeatureOffset, splitIdx))
                val rightChildStats = (binAggregates.getImpurityCalculator(nodeFeatureOffset, maxFuzzySet),
                  binAggregates.getFreqStats(nodeFeatureOffset, maxFuzzySet))
                rightChildStats._1.subtract(leftChildStats._1) //update uStats
                rightChildStats._2.indices.foreach(i => rightChildStats._2(i) -= leftChildStats._2(i)) //update fStats
                predictWithImpurity = Some(predictWithImpurity.getOrElse(
                  calculatePredictImpurity(leftChildStats, rightChildStats)))
                val gainStats = calculateGainForSplit(leftChildStats,
                  rightChildStats, binAggregates.metadata, predictWithImpurity.get._2)
                (splitIdx, gainStats)
              }.maxBy(_._2.gain)
          val fuzzySetsRange = (minFuzzySet to bestFeatureSplitIndex).toList
          val bestFeatureSplit = new BinarySplit(featureIndex,
            fuzzySetsRange.map(_.toInt), fuzzySetsRange.map(valueIndex =>
              binAggregates.metadata.featureIndexAndIndexInPartitionToId((featureIndex, valueIndex))))
          (bestFeatureSplit, bestFeatureGainStats)
        } else if (binAggregates.metadata.isUnordered(featureIndex)) {
          // Unordered categorical feature
          val (leftChildOffset, rightChildOffset) =
            binAggregates.getLeftRightFeatureOffsets(featureIndex)
          val (bestFeatureSplitIndex, bestFeatureGainStats) =
            (0 until numSplits).map { splitIndex =>
              if (nodeFeatureSplitsHistory == null ||
                (splits(featureIndex)(splitIndex).categoryIndexes.toSet.subsetOf(nodeFeatureSplitsHistory)
                  && splits(featureIndex)(splitIndex).categoryIndexes.toSet.size != nodeFeatureSplitsHistory.size)){
                val leftChildStats = (binAggregates.getImpurityCalculator(leftChildOffset, splitIndex),
                  binAggregates.getFreqStats(leftChildOffset, splitIndex))
                val rightChildStats = (binAggregates.getImpurityCalculator(rightChildOffset, splitIndex),
                  binAggregates.getFreqStats(rightChildOffset, splitIndex))
                predictWithImpurity = Some(predictWithImpurity.getOrElse(
                  calculatePredictImpurity(leftChildStats, rightChildStats)))
                val gainStats = calculateGainForSplit(leftChildStats,
                  rightChildStats, binAggregates.metadata, predictWithImpurity.get._2)
                (splitIndex, gainStats)
              }else{
                (splitIndex, BinaryInformationGainStats.invalidInformationGainStats)
              }
            }.maxBy(_._2.gain)
          (splits(featureIndex)(bestFeatureSplitIndex), bestFeatureGainStats)
        }else{
          // Ordered categorical feature
          /*
           * Continuous and categorical feature are handled in the same way
           * (Note that is true because we consider only strong fuzzy partitions)
           */
          val nodeFeatureOffset = binAggregates.getFeatureOffset(featureIndex)
          val numBins = binAggregates.metadata.numFeatureBin(featureIndex)
          /* Each bin is one category (feature value).
           * The bins are ordered based on centroids, and this ordering determines which
           * splits are considered.  (With L categories, we consider L - 1 possible splits.)
           * Feature value can be a fuzzy set in case of continuous feature or one possible value
           * of categorical features
           *
           * centroidsForCategories is a list: (category, centroid)
           */
          val centroidsForCategories = if (binAggregates.metadata.isMultiClass){
            // In multiclass classification,
            // the bins are ordered by the impurity of their corresponding labels.
            (0 until numBins).map { case binIdx =>
              val categoryStats = binAggregates.getImpurityCalculator(nodeFeatureOffset, binIdx)
              val centroid = if (categoryStats.count != 0) {
                categoryStats.calculate()
              } else {
                Double.MaxValue
              }

              (binIdx, centroid)
            }
          }else{
            // In binary classification,
            // the bins are ordered by the centroid of their corresponding labels.
            (0 until numBins).map{ case binIndex =>
              val categoryStats = binAggregates.getImpurityCalculator(nodeFeatureOffset, binIndex)
              val centroid = if (categoryStats.count != 0) {
                categoryStats.predict
              } else {
                Double.MaxValue
              }
              (binIndex, centroid)
            }
          }

          // Bins sorted by centroids
          val categoriesSortedByCentroid = centroidsForCategories.toList.sortBy(_._2)

          // Cumulative sum (scanLeft) of bin statistics.
          // Afterwards, binAggregates for a bin is the sum of aggregates for
          // that bin + all preceding bins.
          var splitIndex = 0
          while (splitIndex < numSplits) {
            val currentCategory = categoriesSortedByCentroid(splitIndex)._1
            val nextCategory = categoriesSortedByCentroid(splitIndex + 1)._1
            binAggregates.mergeForFeature(nodeFeatureOffset, nextCategory, currentCategory)
            splitIndex += 1
          }
          // lastCategory = index of bin with total aggregates for this (node, feature)
          val lastCategory = categoriesSortedByCentroid.last._1
          // Find best split
          val (bestFeatureSplitIndex, bestFeatureGainStats) =
            (0 until numSplits).map { splitIndex =>
              val featureValue = categoriesSortedByCentroid(splitIndex)._1
              if (nodeFeatureSplitsHistory == null ||
                (Set(featureValue).subsetOf(nodeFeatureSplitsHistory)
                  && nodeFeatureSplitsHistory.size != 1)){
                val leftChildStats =
                  (binAggregates.getImpurityCalculator(nodeFeatureOffset, featureValue),
                    binAggregates.getFreqStats(nodeFeatureOffset, featureValue))
                val rightChildStats =
                  (binAggregates.getImpurityCalculator(nodeFeatureOffset, lastCategory),
                    binAggregates.getFreqStats(nodeFeatureOffset, lastCategory))
                // Subtract the stats from right child stats with left child stats
                // for both impurity calculator and frequency stats
                rightChildStats._1.subtract(leftChildStats._1)
                rightChildStats._2.indices.foreach { i =>
                  rightChildStats._2(i) -= leftChildStats._2(i)
                }
                predictWithImpurity = Some(predictWithImpurity.getOrElse(
                  calculatePredictImpurity(leftChildStats, rightChildStats)))
                val gainStats = calculateGainForSplit(leftChildStats,
                  rightChildStats, binAggregates.metadata, predictWithImpurity.get._2)
                (splitIndex, gainStats)
              }else{
                (splitIndex, BinaryInformationGainStats.invalidInformationGainStats)
              }
            }.maxBy(_._2.gain)

          val categoriesForSplit =
            categoriesSortedByCentroid.map(_._1.toDouble).slice(0, bestFeatureSplitIndex + 1)
          val bestFeatureSplit = new BinarySplit(featureIndex,
            categoriesForSplit.map(_.toInt), categoriesForSplit.map(valueIndex =>
              binAggregates.metadata.featureIndexAndIndexInPartitionToId((featureIndex, valueIndex.toInt))))
          (bestFeatureSplit, bestFeatureGainStats)
        }
      }
    }.maxBy(_._2.gain)

    (bestSplit, bestSplitStats, predictWithImpurity.get._1)
  }

  /**
    * Returns splits and bins for decision tree calculation.
    * Continuous and categorical features are handled in the same way.
    *  For each feature, splits are handled in 2 ways:
    *    (a) "unordered features"
    *        For multi-class classification with a low-arity feature
    *        (i.e., if isMulticlass && isSpaceSufficientForAllCategoricalSplits),
    *        the feature is split based on subsets of categories.
    *    (b) "ordered features"
    *        For binary classification and for multiclass classification
    *        with a high-arity feature, there is one bin per category.
    *
    * @param metadata Learning and dataset metadata
    * @return An array of [[iet.unipi.bigdatamining.classification.tree.model.binary.BinarySplit]].
    *          of size (numFeatures, numSplits).
    */
  protected[tree] def findSplits(
     metadata: FuzzyDecisionTreeMetadata): Array[Array[BinarySplit]] = {

    val numFeatures = metadata.numFeatures
    val splits = new Array[Array[BinarySplit]](numFeatures)
    // Find all splits.
    // Iterate over all features.
    var featureIndex = 0
    while (featureIndex < numFeatures) {
      val numBins = metadata.numFeatureBin(featureIndex)
      val numSplits = metadata.numSplits(featureIndex)
      // Arity of feature
      val  featureArity =
        if (metadata.isCategorical(featureIndex))
          metadata.categoricalFeaturesArity(featureIndex)
        else
          metadata.fuzzySetFeatures(featureIndex).length

      if (metadata.isUnordered(featureIndex)) {
        // Unordered features
        //   2^(maxFeatureValue - 1) - 1 combinations
        splits(featureIndex) = new Array[BinarySplit](numSplits)
        var splitIndex = 0
        while (splitIndex < numSplits) {
          val categories: List[Double] =
            extractMultiClassCategories(splitIndex + 1, featureArity)
          splits(featureIndex)(splitIndex) =
            new BinarySplit(featureIndex, categories.map(_.toInt),
              categories.map(valueIndex =>
                metadata.featureIndexAndIndexInPartitionToId((featureIndex, valueIndex.toInt))))
          splitIndex += 1
        }
      }else{
        // Ordered features
        //   Bins correspond to feature values, so we do not need to compute splits or bins
        //   beforehand.  Splits are constructed as needed during training.
        splits(featureIndex) = new Array[BinarySplit](0)
      }
      featureIndex += 1
    }
    splits
  }

  /**
    * Nested method to extract list of eligible categories given an index. It extracts the
    * position of ones in a binary representation of the input. If binary
    * representation of an number is 01101 (13), the output list should (3.0, 2.0,
    * 0.0). The maxFeatureValue depict the number of rightmost digits that will be tested for ones.
    *
    * @param input used as binary representation
    * @param maxFeatureValue feature arity
    * @return a list of eligible categories given an index
    */
  private[tree] def extractMultiClassCategories(
     input: Int,
     maxFeatureValue: Int): List[Double] = {
    var categories = List[Double]()
    var j = 0
    var bitShiftedInput = input
    while (j < maxFeatureValue) {
      if (bitShiftedInput % 2 != 0) {
        // updating the list of categories.
        categories = j.toDouble :: categories
      }
      // Right shift by one
      bitShiftedInput = bitShiftedInput >> 1
      j += 1
    }
    categories
  }

  /**
    * Pull nodes off of the queue, and collect a group of nodes to be split on this iteration.
    * This tracks the memory usage for aggregates and stops adding nodes when too much memory
    * will be needed; this allows an adaptive number of nodes since different nodes may require
    * different amounts of memory (if featureSubsetStrategy is not "all").
    *
    * @param nodeQueue  Queue of nodes to split.
    * @param maxMemoryUsage  Bound on size of aggregate statistics.
    * @param metadata of Fuzzy Decision Tree
    * @return an array of nodes to be split in the next iteration
    */
  private[tree] def selectNodesToSplit(
                                        nodeQueue: mutable.Queue[BinaryNode],
                                        maxMemoryUsage: Long,
                                        metadata: FuzzyDecisionTreeMetadata): Array[BinaryNode] = {
    // Collect some nodes to split:
    //  nodesForGroup(treeIndex) = nodes to split
    val mutableNodesForGroup = new mutable.ArrayBuffer[BinaryNode]()
    var memUsage = 0L
    while (nodeQueue.nonEmpty && memUsage < maxMemoryUsage) {
      val node = nodeQueue.head
      // Check if enough memory remains to add this node to the group.
      val nodeMemUsage = metadata.numFeatureBin.values.sum * metadata.numClasses * 8L
      if (memUsage + nodeMemUsage <= maxMemoryUsage) {
        nodeQueue.dequeue()
        mutableNodesForGroup += node
      }
      memUsage += nodeMemUsage
    }
    // Convert mutable array to immutable one.
    mutableNodesForGroup.toArray
  }

}