package iet.unipi.bigdatamining.classification.tree

import iet.unipi.bigdatamining.classification.tree.configuration.FDTStrategy
import iet.unipi.bigdatamining.classification.tree.impl.{FuzzyDecisionTreeMetadata, TreePoint}
import iet.unipi.bigdatamining.classification.tree.model.{FuzzyDecisionTreeModel, PredictStats}
import iet.unipi.bigdatamining.classification.tree.model.multi.{MultiInformationGainStats, MultiNode}
import iet.unipi.bigdatamining.classification.tree.impl.FDTStatsAggregator
import iet.unipi.bigdatamining.classification.tree.impurity.FuzzyImpurityCalculator
import iet.unipi.bigdatamining.fuzzy.fuzzyset.SingletonFuzzySet
import iet.unipi.bigdatamining.fuzzy.tnorm.TNorm
import org.apache.spark.Logging
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.collection.JavaConverters._
import scala.collection.mutable

/**
  * A Companion Class that implements a Fuzzy Multi Decision Tree learning algorithm for classification.
  * It supports both continuous and categorical features.
  *
  * @param fdtStrategy The configuration parameters for the Fuzzy Multi Decision Tree algorithm
  * @since 1.0
to  */
class FuzzyMultiDecisionTree (private val fdtStrategy: FDTStrategy) extends Serializable with Logging{

  fdtStrategy.assertValid()

  /**
    * Method to train a decision tree model over an RDD
    * @param input Training data: RDD of [[org.apache.spark.mllib.regression.LabeledPoint]]
    * @return Decision Tree Model that can be used for prediction
    * @since 1.0
    */
  def run(input: RDD[LabeledPoint]): FuzzyDecisionTreeModel = {
    log.info(s"Building metadata")

    // Builds metadata from input parameters
    val metadata = FuzzyDecisionTreeMetadata.buildMetadata(input, fdtStrategy)

    log.info(s"Converting dataset")

    /*
     *  Differently from DecisionTree of MLlib of Spark,
     *  we don't find the splits and the corresponding bins (necessary step
     *  mainly for continuous features).
     *  We recall that for continuous features, a fuzzy partition has been already
     *  defined. Thus, since we don't use binary splits as well as the DecisionTree,
     *  there is a one-to-one correspondence between the splits and the bins.
     *  Indeed each fuzzy set or categorical value is a bin and each node contains
     *  only one of them
     */
    // Bin feature values (TreePoint representation).
    // Cache input RDD for speedup during multiple passes.
    val treeInput = TreePoint.convertToTreeRDD(input, metadata).cache()

    // depth of the decision tree
    val maxDepth = fdtStrategy.maxDepth
    require(maxDepth <= 30,
      s"DecisionTree currently only supports maxDepth <= 30, but was given maxDepth = $maxDepth.")

    // Max memory usage for aggregates (Calculate more precisely)
    val maxMemoryUsage = fdtStrategy.maxMemoryInMB * 1024L * 1024L
    val maxMemoryPerNode = metadata.numFeatureBin.values.sum * metadata.numClasses * 8L

    require(maxMemoryPerNode <= maxMemoryUsage,
      s"RandomForest/DecisionTree given maxMemoryInMB = ${fdtStrategy.maxMemoryInMB}," +
        s" which is too small for the given features." +
        s"  Minimum value = ${maxMemoryPerNode / (1024L * 1024L)}")

    log.info(s"Building Fuzzy Multi Decision Tree")

    /*
     * The main idea here is to perform group-wise training of the fuzzy decision tree nodes thus
     * reducing the passes over the data from (# nodes) to (# nodes / maxNumberOfNodesPerGroup).
     * Each data sample is handled by a particular node (or it reaches a leaf and is not used
     * in lower levels).
     */
    // Allocate and queue root nodes.
    val root = MultiNode.emptyNode(id = 1)
    // FIFO queue of nodes to train: (treeIndex, node)
    val nodeQueue = new mutable.Queue[MultiNode]()
    nodeQueue.enqueue(root)
    while (nodeQueue.nonEmpty) {
      // Collect some nodes to split, and choose features for each node (if subsampling).
      // Each group of nodes may come from one or multiple levels.
      val nodesToSplit = FuzzyMultiDecisionTree.selectNodesToSplit(nodeQueue, maxMemoryUsage, metadata)
      // Sanity check (should never occur):
      assert(nodesToSplit.length > 0,
        s"Fuzzy Multi Decision Tree selected empty nodesForGroup. Error for unknown reason.")

      log.info(s"Splitting ${nodesToSplit.length} nodes")

      // Choose node splits, and enqueue new nodes as needed.
      FuzzyMultiDecisionTree.findBestSplits(treeInput, metadata, root, MultiNode.getMaxId(root)+1,
        nodesToSplit, nodeQueue)
    }

    treeInput.unpersist()

    log.info(s"Fuzzy Multi Decision Tree built")

    val idToFeatureIdFuzzySet = metadata.idToFeatureIndexAndIndexInPartition.map{ case (k,v) =>
      if (metadata.isContinuous(v._1))
        (k, (v._1, metadata.fuzzySetFeatures(v._1)(v._2)))
      else
        (k, (v._1, new SingletonFuzzySet(v._2)))
    }

    log.info(s"Returning the Fuzzy Multi Decision Tree model")

    // Create and return the model
    new FuzzyDecisionTreeModel(root, metadata.tNorm, idToFeatureIdFuzzySet)
  }
}

/**
  * Companion object of FuzzyMultiDecisionTree
  */
object FuzzyMultiDecisionTree {

  private final def SPLIT_TYPE: String = "multi_split"

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
             maxDepth: Int = 5,
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
    new FuzzyMultiDecisionTree(strategy).run(input)
  }

  /**
    * Java-friendly API for [[iet.unipi.bigdatamining.classification.tree.FuzzyMultiDecisionTree#train]]
    */
  def trainFromJava(
                     input: JavaRDD[LabeledPoint],
                     impurity: java.lang.String = "fuzzy_entropy",
                     tNorm: java.lang.String = "Product",
                     maxDepth: java.lang.Integer = 5,
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
      (0 until values.size()).foreach{ i =>
        cutPoints(i) = values.get(i)
      }
      scalaThresholdsFeatureInfo(key.intValue()) = cutPoints
    }
    train(input.rdd, impurity, tNorm, maxDepth.intValue(), maxBins.intValue(), numClasses.intValue(),
      categoricalFeaturesInfo.asInstanceOf[java.util.Map[Int, Int]].asScala.toMap,
      scalaThresholdsFeatureInfo.toMap, minInstancesPerNode.intValue(),
      minFuzzyInstancesPerNode.doubleValue(), minImpurityRatioPerNode.doubleValue(),
      minInfoGain.doubleValue(), subsamplingRate.doubleValue())
  }

  /**
    * Get the node indexes and the activation degree corresponding to this data point.
    * This function mimics prediction, passing an example from the root node down to a leaf
    * or node not split yet; that indexes of each fired node with the associated
    * activation degree are returned.
    *
    * @param node Node in tree from which to classify the given data point.
    * @param binnedFeatures Binned feature vector for data point.
    * @param tNorm for computing the activation degree
    * @param currAD current activation degree
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
                             node: MultiNode,
                             binnedFeatures: Array[Array[(Int, Double)]],
                             tNorm: TNorm,
                             currAD: Double = 1): Iterable[(Int,Double)] = {
    if (node.isLeaf || node.splitFeatureIndex < 0/*.isEmpty*/) {
      // Node is either leaf, or has not been split yet.
      List[(Int,Double)]((node.id, currAD))
    } else {
      // Continuous or Categorical features are processed in the same way
      //      val featureIndex = node.splitFeatureIndex.get.feature
      val firedChildrenIds = binnedFeatures(node.splitFeatureIndex/*featureIndex*/).toMap
      var childrenIds = Iterable.empty[(Int, Double)]
      firedChildrenIds.foreach { case (firedChildId, firedMD) =>
        childrenIds = childrenIds ++ getFiredNodes(node.children.apply(firedChildId).get,
          binnedFeatures, tNorm, tNorm.calculate(currAD, firedMD))
      }
      childrenIds
    }
  }

  /**
    * Helper for binSeqOp for updating the sufficient statistics for each feature.
    *
    * @param agg  Array storing aggregate calculation, with a set of sufficient statistics for
    *             each (feature, bin).
    * @param treePoint  Data point being aggregated.
    * @param uAs activation degree
    */
  private def binSeqOpAggregatorHelper(
                                        agg: FDTStatsAggregator,
                                        treePoint: TreePoint,
                                        uAs: Double): Unit = {
    // Get the label of the point
    val label = treePoint.label
    // Iterate over all features.
    val numFeatures = agg.metadata.numFeatures
    (0 until numFeatures).foreach {featureIndex =>
      val binIndexes = treePoint.binnedFeatures(featureIndex)
      binIndexes.foreach{ binIndex =>
        agg.updateStats(featureIndex, binIndex._1, label, binIndex._2, uAs)
      }
    }
  }

  /**
    * Given a group of nodes, this method finds the best split for each node.
    *
    * @param input Training data: RDD of [[org.apache.spark.mllib.tree.impl.TreePoint]]
    * @param metadata Learning and dataset metadata
    * @param root Root node for each tree.  Used for matching instances with nodes.
    * @param startNodeIndex id of the node on shiwch start to count (different from each iteration)
    * @param nodesToSplit Mapping: treeIndex --> nodes to be split in tree
    * @param nodeQueue  Queue of nodes to split, with values (treeIndex, node).
    *                   Updated with new non-leaf nodes which are created.
    */
  private[tree] def findBestSplits(
                                    input: RDD[TreePoint],
                                    metadata: FuzzyDecisionTreeMetadata,
                                    root: MultiNode,
                                    startNodeIndex: Int,
                                    nodesToSplit: Array[MultiNode],
                                    nodeQueue: mutable.Queue[MultiNode]): Unit = {

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
     *
     */

    // numNodes:  Number of nodes in this group
    val numNodes = nodesToSplit.length
    // Reassign to a var
    var startNodeIdx = startNodeIndex
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
                      uAs: Double): Unit = {
      if (nodeIndexInGroup >= 0) {
        binSeqOpAggregatorHelper(agg(nodeIndexInGroup), point, uAs)
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
      * @param point   Data point being aggregated.
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

    val nodeToBestSplits = partitionAggregates.reduceByKey((a,b) => a.merge(b))
      .map{ case (nodeIndexInGroup, aggStats) =>
        // Find best split for each node
        val (featureId: Int, stats: MultiInformationGainStats, predict: PredictStats) =
          binsToBestSplit(aggStats, nodesToSplit(nodeIndexInGroup))
        (nodeIndexInGroup, (featureId, stats, predict))
      }.collectAsMap()

    nodesToSplit.foreach{ node =>
      val nodeId = node.id
      val nodeIndexInGroup = nodeIdToNodeIndexInGroup(nodeId)
      val (featureId: Int, gainStats: MultiInformationGainStats, predictStats: PredictStats) =
        nodeToBestSplits(nodeIndexInGroup)

      // Extract info for this node. Create children if not leaf.
      val isLeaf = ((gainStats.gain <= 0D) || (node.level == metadata.maxDepth)
        || (predictStats.getMaxLabelPredictionRatio >= metadata.minImpurityRatioPerNode))
      node.predictStats = predictStats
      node.isLeaf = isLeaf
      node.gainStats = Some(gainStats)
      node.impurity = gainStats.impurity

      if (!isLeaf){
        val numBins = metadata.numFeatureBin.get(featureId).get
        // Create split for the node
        node.splitFeatureIndex = featureId
        val childLevel = node.level + 1
        val childIsLeaf = childLevel == metadata.maxDepth
        val childrenAreLeaves = gainStats.childrenImpurity.indices.map{ i =>
          ((gainStats.childrenImpurity(i) <= 0D) || childIsLeaf
            || (gainStats.childrenPredict(i).totalFreq <= metadata.minInstancesPerNode)
            || (gainStats.childrenPredict(i).totalU <= metadata.minFuzzyInstancesPerNode))
        }
        node.children = (0 until numBins).map{ binIndex =>
          Some(MultiNode(startNodeIdx + binIndex,
            metadata.featureIndexAndIndexInPartitionToId.get((featureId, binIndex)).get,
            childLevel, gainStats.childrenImpurity(binIndex),
            childrenAreLeaves(binIndex), gainStats.childrenPredict(binIndex),
            node.featureHistory ++ List(featureId).toIterator))
        }.toArray

        // Update start node id for current tree
        startNodeIdx += numBins

        // Enqueue each child if it is not leaf
        node.children.foreach{ childNode =>
          if (!childNode.get.isLeaf)
            nodeQueue.enqueue(childNode.get)
        }
      }

    }

  }

  /**
    * Calculate the information gain for a given (feature, split) based upon left/right aggregates.
    *
    * @param childrenImpurityCalculatorsFreqStats fuzzy impurity calculator for each child of the split
    * @param metadata of the fuzzy decision tree to be trained
    * @param impurity of the node to be split
    * @return information gain and statistics for split
    */
  private def calculateGain(
                             childrenImpurityCalculatorsFreqStats: Array[(FuzzyImpurityCalculator, Array[Double])],
                             metadata: FuzzyDecisionTreeMetadata,
                             impurity: Double): MultiInformationGainStats = {
    /*
     *  Calculate Weighted Fuzzy Impurity WFent(Split;S)
     *  Note that since the partitions is strong, totalCount
     *  is equal to the frequency of the examples and represent
     *  the cardinality of D
     */
    val childrenCardinality = childrenImpurityCalculatorsFreqStats.map(_._1.count)
    val childrenFuzzyImpurities = childrenImpurityCalculatorsFreqStats.map(_._1.calculate())
    val totalCardinality = childrenCardinality.sum.toDouble
    val wFuzzyImpurity = childrenFuzzyImpurities.indices.aggregate(0D)(
      (currFuzzyImpurity, childIndex) =>
        currFuzzyImpurity + (childrenFuzzyImpurities(childIndex) * childrenCardinality(childIndex)),
      (currFuzzyImpurityI, currFuzzyImpurityJ) => currFuzzyImpurityI+currFuzzyImpurityJ) / totalCardinality

    // Calculate gain (impurity - WFent)
    val gain = impurity - wFuzzyImpurity

    // If information gain doesn't satisfy minimum information gain,
    // then this split is invalid and return invalid information gain stats.
    if (gain < metadata.minInfoGain) {
      return MultiInformationGainStats.invalidInformationGainStats
    }

    val childrenPredicts = childrenImpurityCalculatorsFreqStats.map{ childCalculator =>
      calculatePredict(childCalculator._1.stats, childCalculator._2)
    }

    new MultiInformationGainStats(gain, impurity, childrenFuzzyImpurities, childrenPredicts)
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
    * Calculate predict value for current node, given stats of any split.
    * Note that this function is called only once for each tree and
    * it works only if fuzzy partition is strong.
    *
    * @param childrenFuzzyImpurityCalculatorFreqStats children node aggregates for a split
    * @return predict value and impurity for current node
    */
  private def calculatePredictImpurity(
      childrenFuzzyImpurityCalculatorFreqStats: Array[(FuzzyImpurityCalculator, Array[Double])]): (PredictStats, Double) = {
    // Calculate Impurity
    val parentFuzzyImpurityCalculator = childrenFuzzyImpurityCalculatorFreqStats(0)._1.copy
    var parentFreqStats = childrenFuzzyImpurityCalculatorFreqStats(0)._2.clone
    if (childrenFuzzyImpurityCalculatorFreqStats.length > 1){
      Range(1, childrenFuzzyImpurityCalculatorFreqStats.length).foreach{index =>
        parentFuzzyImpurityCalculator.add(childrenFuzzyImpurityCalculatorFreqStats(index)._1)
        parentFreqStats = (parentFreqStats, childrenFuzzyImpurityCalculatorFreqStats(index)._2).zipped.map(_ + _)
      }
    }
    val impurity = parentFuzzyImpurityCalculator.calculate()
    // Calculate PredictStats
    val predict = calculatePredict(parentFuzzyImpurityCalculator.stats, parentFreqStats)
    (predict, impurity)
  }

  /**
    * Find the best split for a node.
    *
    * @param binAggregates Bin statistics.
    * @param node on which calculate the best splits
    * @return tuple for best split: (Split, information gain, prediction at node)
    */
  private def binsToBestSplit(
                               binAggregates: FDTStatsAggregator,
                               node: MultiNode): (Int, MultiInformationGainStats, PredictStats) = {
    // Calculate predict and impurity if current node is top node
    var predictWithImpurity = if (node.isRoot) { None } else {
      Some((node.predictStats, node.impurity))
    }
    // For each feature calculate the gain and select the best one
    val (bestFeatureIndex, bestSplitStats) = {
      (0 until binAggregates.metadata.numFeatures).map { featureIndex =>
        if (node.featureHistory.contains(featureIndex)){
          (featureIndex, MultiInformationGainStats.invalidInformationGainStats)
        }else{
          val numBins = binAggregates.metadata.numFeatureBin(featureIndex)
          val childrenStats = new Array[(FuzzyImpurityCalculator, Array[Double])](numBins)
          val nodeFeatureOffset = binAggregates.getFeatureOffset(featureIndex)
          /*
           * Continuous and categorical feature are handled in the same way
           * (Note that is true because we consider only strong fuzzy partitions) 
          */
          (0 until numBins).foreach { case binIdx =>
            childrenStats(binIdx) = (binAggregates.getImpurityCalculator(nodeFeatureOffset, binIdx),
              binAggregates.getFreqStats(nodeFeatureOffset, binIdx))
          }
          predictWithImpurity = Some(predictWithImpurity.getOrElse(
            calculatePredictImpurity(childrenStats)))
          val gainStats = calculateGain(childrenStats,
            binAggregates.metadata, predictWithImpurity.get._2)

          (featureIndex, gainStats)
        }
      }.maxBy(_._2.gain) // Get the best feature according the maximum gain obtained 

    }

    (bestFeatureIndex, bestSplitStats, predictWithImpurity.get._1)
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
    * @return  an array of nodes to be split in the next iteration
    */
  private[tree] def selectNodesToSplit(
                                        nodeQueue: mutable.Queue[MultiNode],
                                        maxMemoryUsage: Long,
                                        metadata: FuzzyDecisionTreeMetadata): Array[MultiNode] = {
    // Collect some nodes to split:
    // nodesForGroup(treeIndex) = nodes to split
    val mutableNodesForGroup = new mutable.ArrayBuffer[MultiNode]()
    var memUsage: Long = 0L
    while (nodeQueue.nonEmpty && memUsage < maxMemoryUsage) {
      val node: MultiNode = nodeQueue.head
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