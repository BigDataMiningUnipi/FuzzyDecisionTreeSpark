package iet.unipi.bigdatamining.classification.tree

import java.text.DecimalFormat
import java.math.RoundingMode

import iet.unipi.bigdatamining.classification.tree.configuration.FDTStrategy
import iet.unipi.bigdatamining.classification.tree.impl.FuzzyDecisionTreeMetadata
import iet.unipi.bigdatamining.classification.tree.model.{FuzzyDecisionTreeModel, PredictStats}
import iet.unipi.bigdatamining.fuzzy.fuzzyset.{SingletonFuzzySet, TriangularFuzzySet}
import com.holdenkarau.spark.testing.SharedSparkContext
import iet.unipi.bigdatamining.fuzzy.tnorm.{MinTNorm, ProductTNorm}
import iet.unipi.bigdatamining.classification.tree.model.multi.MultiNode
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.Node
import org.junit.runner.RunWith
import org.scalatest.Matchers._
import org.scalatest.FunSuite
import org.scalatest.junit.JUnitRunner

import scala.collection.immutable.Map
import scala.io.Source


@RunWith(classOf[JUnitRunner])
class FuzzyDecisionTreeSuite extends FunSuite with SharedSparkContext {

  //******************************************************************
  //*    1. Test for building metadata, fuzzy sets and FDT nodes
  //******************************************************************

  //Test 1.1
  test("Test 1.1: Classification with continuous features; testing metadata") {
    val arr = FuzzyDecisionTreeSuite.generateOrderedLabeledPointsWithLabel1
    arr.length should be (1000)
    val rdd = sc.parallelize(arr)
    val strategy = new FDTStrategy(
      splitType = "binary_split",
      impurity = "fuzzy_entropy",
      tNorm = "product",
      maxDepth = 3,
      numClasses = 2,
      maxBins = 100,
      thresholdsFeatureInfo = Map(0 -> Array(100D, 200D, 300D, 400D, 500D, 600D, 700D, 800D, 900D)))
    val metadata = FuzzyDecisionTreeMetadata.buildMetadata(rdd, strategy)
    metadata.isUnordered(featureIndex = 0) should be (false)
  }

  // Test 1.2
  test("Test 1.2: Binary classification with binary (ordered) categorical features; testing metadata") {
    val arr = FuzzyDecisionTreeSuite.generateCategoricalDataPoints
    arr.length should be (1000)
    val rdd = sc.parallelize(arr)
    val strategy = new FDTStrategy(
      splitType = "binary_split",
      impurity = "fuzzy_entropy",
      tNorm = "product",
      maxDepth = 2,
      numClasses = 2,
      maxBins = 100,
      categoricalFeaturesInfo = Map(0 -> 2, 1 -> 2))

    val metadata = FuzzyDecisionTreeMetadata.buildMetadata(rdd, strategy)
    metadata.isUnordered(featureIndex = 0) should be (false)
    metadata.isUnordered(featureIndex = 1) should be (false)
    // Pre-computed splits for ordered categorical features (arity - 1)
    metadata.numSplits(0) should be (1)
    metadata.numSplits(1) should be (1)
  }

  // Test 1.3
  test("Test 1.3: Multiclass classification with unordered categorical features; testing metadata") {
    val arr = FuzzyDecisionTreeSuite.generateCategoricalDataPoints
    arr.length should be (1000)
    val rdd = sc.parallelize(arr)
    val strategy = new FDTStrategy(
      splitType = "binary_split",
      impurity = "fuzzy_entropy",
      tNorm = "product",
      maxDepth = 2,
      numClasses = 100,
      maxBins = 100,
      categoricalFeaturesInfo = Map(0 -> 3, 1 -> 3))

    val metadata = FuzzyDecisionTreeMetadata.buildMetadata(rdd, strategy)
    metadata.isUnordered(featureIndex = 0) should be (true)
    metadata.isUnordered(featureIndex = 1) should be (true)
  }

  // Test 1.4
  test("Test 1.4: Multiclass classification with ordered categorical features; testing metadata") {
    val arr = FuzzyDecisionTreeSuite.generateCategoricalDataPoints
    arr.length should be (1000)
    val rdd = sc.parallelize(arr)
    val strategy = new FDTStrategy(
      splitType = "binary_split",
      impurity = "fuzzy_entropy",
      tNorm = "product",
      maxDepth = 2,
      numClasses = 100,
      maxBins = 100,
      categoricalFeaturesInfo = Map(0 -> 10, 1 -> 10))
    // 2^(10-1) - 1 > 100, so categorical features will be ordered

    val metadata = FuzzyDecisionTreeMetadata.buildMetadata(rdd, strategy)
    metadata.isUnordered(featureIndex = 0) should be (false)
    metadata.isUnordered(featureIndex = 1) should be (false)
    // Pre-computed splits for ordered categorical features (arity - 1)
    metadata.numSplits(0) should be (9)
    metadata.numSplits(1) should be (9)
  }

  // Test 1.5
  test("Test 1.5: Extract categories from a number for multiclass classification") {
    val l = FuzzyBinaryDecisionTree.extractMultiClassCategories(13, 10)
    l.size should be (3)
    List(3.0, 2.0, 0.0) should be (l)
  }

  // Test 1.6
  test("Test 1.6: Create Strong Fuzzy Partition: Triangular fuzzy sets") {
    val arr = Array(1D, 2D, 3D, 4D, 5D)
    TriangularFuzzySet.createFuzzySets(arr.take(2)).length should be (2)

    val part = TriangularFuzzySet.createFuzzySets(arr)
    part.length should be (5)

    // Check first Triangular Fuzzy set
    part.head.left should be (Double.MinValue)
    part.head.right should be (arr(1))
    part.head.membershipDegree(arr(0)) should be (1D)
    part.head.membershipDegree(arr(1)) should be (0D)
    part.head.isInSupport(-0.5) should be (true)
    part.head.isInSupport(arr(3)) should be (false)

    // Check last Triangular Fuzzy Set
    part.last.left should be (arr(part.length-2))
    part.last.right should be (Double.MaxValue)
    part.last.membershipDegree(arr(part.length-1)) should be (1D)
    part.last.membershipDegree(arr(3)) should be (0D)
    part.last.isInSupport(5.5) should be (true)
    part.last.isInSupport(arr(2)) should be (false)

    // Check all Triangular Fuzzy Sets in the middle
    for (i <- 1 until arr.length-1) {
      part(i).left should be(arr(i-1))
      part(i).right should be(arr(i+1))
      part(i).membershipDegree(arr(i)) should be (1D)
      part(i).membershipDegree(arr(0)) should be (0D)
      part(i).isInSupport((arr(i)+arr(i-1))/2) should be (true)
      part(i).isInSupport(-1D) should be (false)
    }

  }

  // Test 1.7
  test("Test 1.7: Create Strong Fuzzy Partition: Singleton Fuzzy Sets") {
    val arr = Array(1D, 2D, 3D, 4D, 5D)
    SingletonFuzzySet.createFuzzySets(arr.take(2)).length should be (2)

    val part = SingletonFuzzySet.createFuzzySets(scala.util.Random.shuffle(arr.toBuffer).toArray)
    part.length should be (5)

    // Check all Singleton Fuzzy Sets
    arr.indices.foreach { case i =>
      part(i).left should be(arr(i))
      part(i).right should be(arr(i))
      part(i).membershipDegree(arr(i)) should be (1D)
      part(i).membershipDegree(-1D) should be (0D)
      part(i).isInSupport(arr(i)) should be (true)
      part(i).isInSupport(-1D) should be (false)
    }

  }

  // Test 1.8
  test("Test 1.8: Check MaxID for Multi Nodes (i.e. node of Fuzzy Multi Decision Trees)") {
    val node1 = new MultiNode(1, -1, 0, -1D, false, Array.empty[Int], 1, None,
      PredictStats.empty, Array.empty[Option[MultiNode]])

    val childNodes1 = new Array[Option[MultiNode]](10)
    childNodes1.indices.foreach(i => childNodes1(i) = Option(
      new MultiNode(2+i, 2+i, 1, -1D, false, Array.empty[Int], 1, None,
        PredictStats.empty, Array.empty[Option[MultiNode]])))

    val childNodes2 = new Array[Option[MultiNode]](2)
    childNodes2.indices.foreach(i => childNodes2(i) = Option(
      new MultiNode(12+i, 12+i, 2, -1D, false, Array.empty[Int], 1, None,
        PredictStats.empty, Array.empty[Option[MultiNode]])))

    val childNodes3 = new Array[Option[MultiNode]](1)
    childNodes3.indices.foreach(i => childNodes3(i) = Option(
      new MultiNode(14+i, 14+i, 2, -1D, false, Array.empty[Int], 1, None,
        PredictStats.empty, Array.empty[Option[MultiNode]])))

    val childNodes4 = new Array[Option[MultiNode]](5)
    childNodes4.indices.foreach(i => childNodes4(i) = Option(
      new MultiNode(15+i, 15+i, 2, -1D, false, Array.empty[Int], 1, None,
        PredictStats.empty, Array.empty[Option[MultiNode]])))

    MultiNode.getMaxId(node1) should be (1)

    node1.children = childNodes1
    MultiNode.getMaxId(node1) should be (11)

    node1.children.head.get.children = childNodes2
    MultiNode.getMaxId(node1) should be (13)

    node1.children.last.get.children = childNodes3
    MultiNode.getMaxId(node1) should be (14)

    node1.children(4).get.children = childNodes4
    MultiNode.getMaxId(node1) should be (19)

    MultiNode.getMaxId(node1.children(4).get) should be (19)
    MultiNode.getMaxId(node1.children.last.get) should be (14)
    MultiNode.getMaxId(node1.children(4).get.children(2).get) should be (17)
  }

  //******************************************************************
  //*                 2. Test for building trees
  //******************************************************************

  // Test 2.1: for dataset with only categorical features,
  // FDT and DT of Mllib must be the same
  test("Test 2.1: Multiclass classification with unordered categorical features:" +
    " Fuzzy Binary Decision Tree vs Mllib Decision Tree") {
    val arr = FuzzyDecisionTreeSuite.generateCategoricalDataPoints
    arr.length should be (1000)
    val rdd = sc.parallelize(arr)

    // Run Fuzzy Binary Decision Tree
    val impurity = "fuzzy_entropy"
    val tNorm = "product"
    val maxDepth = 3
    val maxBins = 100
    val numClasses = 2
    val categoricalFeaturesInfo = Map(0 -> 2, 1 -> 2)
    val thresholdsFeatureInfo = Map.empty[Int, Array[Double]]
    val fdtModel = FuzzyBinaryDecisionTree.train(rdd, impurity, tNorm, maxDepth, maxBins, numClasses,
      categoricalFeaturesInfo, thresholdsFeatureInfo)

    // Run Mllib Decision Tree
    val dtModel = DecisionTree.trainClassifier(
      input = rdd,
      numClasses = 2,
      categoricalFeaturesInfo = Map(0 -> 2, 1 -> 2),
      impurity = "entropy",
      maxDepth = 3,
      maxBins = 100)

    // Check Models
    fdtModel.depth should be (dtModel.depth)
    fdtModel.numNodes should be (dtModel.numNodes)
    fdtModel.numLeaves should be (FuzzyDecisionTreeSuite.numLeaves(dtModel.topNode))

    // Evaluate Models
    rdd.foreach(point =>
      fdtModel.predictByMaxValue(point.features) should be (dtModel.predict(point.features)))
  }

  // Test 2.2: for dataset with only categorical features,
  // FDT and DT of Mllib must be the same
  test("Test 2.2: Multiclass classification with ordered categorical features:" +
    " Fuzzy Binary Decision Tree vs Mllib Decision Tree") {
    val arr = FuzzyDecisionTreeSuite.generateCategoricalDataPoints
    arr.length should be (1000)
    val rdd = sc.parallelize(arr)

    // Run Fuzzy Binary Decision Tree
    val impurity = "fuzzy_entropy"
    val tNorm = "product"
    val maxDepth = 3
    val maxBins = 100
    val numClasses = 2
    val categoricalFeaturesInfo = Map(0 -> 10, 1 -> 10)
    val thresholdsFeaturesInfo = Map.empty[Int, Array[Double]]

    // 2^(10-1) - 1 > 100, so categorical features will be ordered
    val fdtModel = FuzzyBinaryDecisionTree.train(rdd, impurity, tNorm, maxDepth, maxBins, numClasses,
      categoricalFeaturesInfo, thresholdsFeaturesInfo)

    // Run Mllib Decision Tree
    val dtModel = DecisionTree.trainClassifier(
      input = rdd,
      numClasses = 2,
      categoricalFeaturesInfo = Map(0 -> 10, 1 -> 10),
      impurity = "entropy",
      maxDepth = 3,
      maxBins = 100)

    // Check Models
    fdtModel.depth should be (dtModel.depth)
    fdtModel.numNodes should be (dtModel.numNodes)
    fdtModel.numLeaves should be (FuzzyDecisionTreeSuite.numLeaves(dtModel.topNode))

    // Evaluate Models
    rdd.foreach(point =>
      fdtModel.predictByMaxValue(point.features) should be (dtModel.predict(point.features)))
  }

  // Test 2.3: each categorical feature has at most 2 values,
  // therefore multi and binary trees must be the same
  test("Test 2.3: Multiclass classification with unordered categorical features:" +
    " Fuzzy Binary Decision Tree vs Fuzzy Multi Decision Tree") {
    val arr = FuzzyDecisionTreeSuite.generateCategoricalDataPoints
    arr.length should be (1000)
    val rdd = sc.parallelize(arr)

    // Set common parameters
    val impurity = "fuzzy_entropy"
    val tNorm = "product"
    val maxDepth = 3
    val maxBins = 100
    val numClasses = 2
    val categoricalFeaturesInfo = Map(0 -> 2, 1 -> 2)
    val thresholdsFeaturesInfo = Map.empty[Int, Array[Double]]

    // Run Fuzzy Binary Decision Tree
    val binaryModel = FuzzyBinaryDecisionTree.train(rdd, impurity, tNorm, maxDepth, maxBins, numClasses,
      categoricalFeaturesInfo, thresholdsFeaturesInfo)

    // Run Fuzzy Multi Decision Tree
    val multiModel = FuzzyMultiDecisionTree.train(rdd, impurity, tNorm, maxDepth, maxBins, numClasses,
      categoricalFeaturesInfo, thresholdsFeaturesInfo)

    // Check Models
    binaryModel.depth should be (multiModel.depth)
    binaryModel.numNodes should be (multiModel.numNodes)
    binaryModel.numLeaves should be (multiModel.numLeaves)

    // Evaluate Models
    rdd.foreach(point =>
      binaryModel.predictByMaxValue(point.features) should be (multiModel.predictByMaxValue(point.features)))
  }

  // Test 2.4: each categorical feature has at most 2 values,
  // therefore multi and binary trees must be the same
  test("Test 2.4: Multiclass classification with ordered categorical features:" +
    " Fuzzy Binary Decision Tree vs Fuzzy Multi Decision Tree") {
    val arr = FuzzyDecisionTreeSuite.generateCategoricalDataPoints
    arr.length should be (1000)
    val rdd = sc.parallelize(arr)

    // Set common parameters for the FDTs
    val impurity = "fuzzy_entropy"
    val tNorm = "product"
    val maxDepth = 3
    val maxBins = 100
    val numClasses = 2
    val thresholdsFeaturesInfo = Map.empty[Int, Array[Double]]

    // Run Fuzzy Binary Decision Tree
    val binaryCategoricalFeaturesInfo = Map(0 -> 10, 1 -> 10)
    // 2^(10-1) - 1 > 100, so categorical features will be ordered
    val binaryModel = FuzzyBinaryDecisionTree.train(rdd, impurity, tNorm, maxDepth, maxBins,
      numClasses, binaryCategoricalFeaturesInfo, thresholdsFeaturesInfo)

    // Run Fuzzy Multi Decision Tree
    val multiCategoricalFeaturesInfo = Map(0 -> 2, 1 -> 2)
    val multiModel = FuzzyMultiDecisionTree.train(rdd, impurity, tNorm, maxDepth, maxBins,
      numClasses, multiCategoricalFeaturesInfo, thresholdsFeaturesInfo)

    // Check Models
    binaryModel.depth should be (multiModel.depth)
    binaryModel.numNodes should be (multiModel.numNodes)
    binaryModel.numLeaves should be (multiModel.numLeaves)

    // Evaluate Models
    rdd.foreach(point =>
      binaryModel.predictByMaxValue(point.features) should be (multiModel.predictByMaxValue(point.features)))
  }

  // Test 2.5: each core of each triangular fuzzy set of each continuous feature
  // is placed in each distinct value of the continuous feature.
  // Therefore multi and binary trees must be the same
  test("Test 2.5: Multiclass classification with continuous features:" +
    " Fuzzy Binary Decision Tree vs Mllib Decision Tree") {
    val arr = FuzzyDecisionTreeSuite.generateCategoricalDataPoints
    arr.length should be (1000)
    val rdd = sc.parallelize(arr)

    // Run Fuzzy Binary Decision Tree
    val impurity = "fuzzy_entropy"
    val tNorm = "product"
    val maxDepth = 3
    val maxBins = 100
    val numClasses = 2
    val categoricalFeaturesInfo = Map.empty[Int, Int]
    val thresholdsFeatureInfo = Map(0 -> Array(0D, 0.5D, 1D), 1 -> Array(0D, 0.5D, 1D))
    val fdtModel = FuzzyBinaryDecisionTree.train(rdd, impurity, tNorm, maxDepth, maxBins,
      numClasses, categoricalFeaturesInfo, thresholdsFeatureInfo)

    // Run Mllib Decision Tree
    val dtModel = DecisionTree.trainClassifier(
      input = rdd,
      numClasses = 2,
      categoricalFeaturesInfo = Map(0 -> 10, 1 -> 10),
      impurity = "entropy",
      maxDepth = 3,
      maxBins = 100)

    // Check Models
    fdtModel.depth should be (dtModel.depth)
    fdtModel.numNodes should be (dtModel.numNodes)
    fdtModel.numLeaves should be (FuzzyDecisionTreeSuite.numLeaves(dtModel.topNode))

    // Evaluate Models
    rdd.foreach(point =>
      fdtModel.predictByMaxValue(point.features) should be (dtModel.predict(point.features)))
  }

  //******************************************************************
  //*               3. Test in real small dataset
  //******************************************************************

  // Test 3.1: Test predictions of FBDT in real iris dataset
  test("Test 3.1: Multiclass classification with continuous features: testing prediction on Fuzzy Binary Decision Trees") {
    val (traArr, tstArr) = FuzzyDecisionTreeSuite.getIrisFromCsvToLabeledPoint
    traArr.length should be (135)
    tstArr.length should be (15)

    // Run Fuzzy Binary Decision Tree
    // We create uniform triangular fuzzy partitions (5 fuzzy sets for each feature)
    val traRdd = sc.parallelize(traArr)
    val impurity = "fuzzy_entropy"
    val tNorm = "product"
    val maxDepth = 5
    val maxBins = 100
    val numClasses = 3
    val categoricalFeaturesInfo = Map.empty[Int, Int]
    val thresholdsFeatureInfo = Map(
      0 -> Array(4.3, 5.2, 6.1, 7.0, 7.9),
      1 -> Array(2.0, 2.6, 3.2, 3.8, 4.4),
      2 -> Array(1.0, 2.475, 3.95, 5.425, 6.9),
      3 -> Array(0.1, 0.7, 1.3, 1.9, 2.6))
    val fdtModel = FuzzyBinaryDecisionTree.train(traRdd, impurity, tNorm, maxDepth, maxBins,
      numClasses, categoricalFeaturesInfo, thresholdsFeatureInfo)

    //////////////////////
    // First prediction //
    //////////////////////
    // The map represents the expected prediction on the first point of the test dataset,
    // i.e four leaves are fired (print model for more details)
    val predictionProduct1 = Map(0 -> 0.999995192184415, 1 -> 4.807815585015025E-6, 2 -> 0D)
    val predictionMin1 = Map(0 -> 0.999995192184415, 1 -> 4.807815585015025E-6, 2 -> 0D)

    // Calculate prediction according different t-norm functions (in this case we have the same results)
    val predictionProductModel1 = fdtModel.topNode.predict(tstArr(0).features, ProductTNorm, fdtModel.idToFeatureIdFuzzySet)
    val predictionMinModel1 = fdtModel.topNode.predict(tstArr(0).features, MinTNorm, fdtModel.idToFeatureIdFuzzySet)

    // Validate predictions
    FuzzyDecisionTreeSuite.validatePredictions(predictionProduct1, predictionProductModel1)
    FuzzyDecisionTreeSuite.validatePredictions(predictionMin1, predictionMinModel1)

    ////////////////////////
    //  Second prediction //
    ////////////////////////
    // 5 leaves are activated
    val feature1 = Vectors.dense(Array(4.525,2.9,1D,1D))
    // The map represents the expected prediction using Product as t-norm function
    val predictionProduct2 = Map(0 -> 0.4010511345, 1 -> 0.59894886550, 2 -> 0D)
    // The map represents the expected prediction using Minimum as t-norm function
    val predictionMin2 = Map(0 -> 0.9210316104, 1 -> 1.3289683896, 2 -> 0D)

    // Calculate prediction of the model with different t-norm functions
    val predictionProductModel2 = fdtModel.topNode.predict(feature1, ProductTNorm, fdtModel.idToFeatureIdFuzzySet)
    val predictionMinModel2 = fdtModel.topNode.predict(feature1, MinTNorm, fdtModel.idToFeatureIdFuzzySet)

    // Validate predictions
    FuzzyDecisionTreeSuite.validatePredictions(predictionProduct2, predictionProductModel2)
    FuzzyDecisionTreeSuite.validatePredictions(predictionMin2, predictionMinModel2)

  }

  // Test 3.2: Test predictions of FMDT in real iris dataset
  test("Test 3.2: Multiclass classification with continuous features: testing prediction on Fuzzy Multi Decision Trees") {
    val (traArr, tstArr) = FuzzyDecisionTreeSuite.getIrisFromCsvToLabeledPoint
    traArr.length should be (135)
    tstArr.length should be (15)

    // Run Fuzzy Binary Decision Tree
    // We create uniform triangular fuzzy partitions (5 fuzzy sets for each feature)
    val traRdd = sc.parallelize(traArr)
    val impurity = "fuzzy_entropy"
    val tNorm = "product"
    val maxDepth = 5
    val maxBins = 100
    val numClasses = 3
    val categoricalFeaturesInfo = Map.empty[Int, Int]
    val thresholdsFeatureInfo = Map(
      0 -> Array(4.3, 5.2, 6.1, 7.0, 7.9),
      1 -> Array(2.0, 2.6, 3.2, 3.8, 4.4),
      2 -> Array(1.0, 2.475, 3.95, 5.425, 6.9),
      3 -> Array(0.1, 0.7, 1.3, 1.9, 2.6))
    val fdtModel = FuzzyMultiDecisionTree.train(traRdd, impurity, tNorm, maxDepth, maxBins,
      numClasses, categoricalFeaturesInfo, thresholdsFeatureInfo)

    //////////////////////
    // First prediction //
    //////////////////////
    // The map represents the expected prediction on the first point of the test dataset,
    // i.e four leaves are fired (print model for more details)
    val predictionProduct1 = Map(0 -> 0.9999955236, 1 -> 0.000004476353383, 2 -> 0D)
    val predictionMin1 = Map(0 -> 1.3332838137, 1 -> 0.00004951965931, 2 -> 0D)

    // Calculate prediction according different t-norm functions (in this case we have the same results)
    val predictionProductModel1 = fdtModel.topNode.predict(tstArr(0).features, ProductTNorm, fdtModel.idToFeatureIdFuzzySet)
    val predictionMinModel1 = fdtModel.topNode.predict(tstArr(0).features, MinTNorm, fdtModel.idToFeatureIdFuzzySet)

    // Validate predictions
    FuzzyDecisionTreeSuite.validatePredictions(predictionProduct1, predictionProductModel1)
    FuzzyDecisionTreeSuite.validatePredictions(predictionMin1, predictionMinModel1)

    ////////////////////////
    //  Second prediction //
    ////////////////////////
    // 5 leaves are activated
    val feature2 = Vectors.dense(Array(4.525,2.9,1D,1D))
    // The map represents the expected prediction using Product as t-norm function
    val predictionProduct2 = Map(0 -> 0.5, 1 -> 0D, 2 -> 0D)
    // The map represents the expected prediction using Minimum as t-norm function (same of product in this case)
    val predictionMin2 = Map(0 -> 0.5, 1 -> 0D, 2 -> 0D)

    // Calculate prediction of the model with different t-norm functions
    val predictionProductModel2 = fdtModel.topNode.predict(feature2, ProductTNorm, fdtModel.idToFeatureIdFuzzySet)
    val predictionMinModel2 = fdtModel.topNode.predict(feature2, MinTNorm, fdtModel.idToFeatureIdFuzzySet)

    // Validate predictions
    FuzzyDecisionTreeSuite.validatePredictions(predictionProduct2, predictionProductModel2)
    FuzzyDecisionTreeSuite.validatePredictions(predictionMin2, predictionMinModel2)

  }

  // Test 3.3: Test FDT in real iris dataset
  test("Test 3.3: Multiclass classification with continuous features: Fuzzy Binary Decision Tree and product tNorm") {
    val (traArr, tstArr) = FuzzyDecisionTreeSuite.getIrisFromCsvToLabeledPoint
    traArr.length should be (135)
    tstArr.length should be (15)

    // Run Fuzzy Binary Decision Tree
    // We create uniform triangular fuzzy partitions (5 fuzzy sets for each feature)
    val traRdd = sc.parallelize(traArr)
    val impurity = "fuzzy_entropy"
    val tNorm = "product"
    val maxDepth = 5
    val maxBins = 100
    val numClasses = 3
    val categoricalFeaturesInfo = Map.empty[Int, Int]
    val thresholdsFeatureInfo = Map(
      0 -> Array(4.3, 5.2, 6.1, 7.0, 7.9),
      1 -> Array(2.0, 2.6, 3.2, 3.8, 4.4),
      2 -> Array(1.0, 2.475, 3.95, 5.425, 6.9),
      3 -> Array(0.1, 0.7, 1.3, 1.9, 2.6))
    val fdtModel = FuzzyBinaryDecisionTree.train(traRdd, impurity, tNorm, maxDepth, maxBins,
      numClasses, categoricalFeaturesInfo, thresholdsFeatureInfo)

    FuzzyDecisionTreeSuite.validateClassifier(fdtModel, traArr.toSeq, 0.96)
    FuzzyDecisionTreeSuite.validateClassifier(fdtModel, tstArr.toSeq, 0.93)

  }

  // Test 3.4: Test FDT in real iris dataset
  test("Test 3.4: Multiclass classification with continuous features: Fuzzy Binary Decision Tree and minimum tNorm") {
    val (traArr, tstArr) = FuzzyDecisionTreeSuite.getIrisFromCsvToLabeledPoint
    traArr.length should be (135)
    tstArr.length should be (15)

    // Run Fuzzy Binary Decision Tree
    // We create uniform triangular fuzzy partitions (5 fuzzy sets for each feature)
    val traRdd = sc.parallelize(traArr)
    val impurity = "fuzzy_entropy"
    val tNorm = "min"
    val maxDepth = 5
    val maxBins = 100
    val numClasses = 3
    val categoricalFeaturesInfo = Map.empty[Int, Int]
    val thresholdsFeatureInfo = Map(
      0 -> Array(4.3, 5.2, 6.1, 7.0, 7.9),
      1 -> Array(2.0, 2.6, 3.2, 3.8, 4.4),
      2 -> Array(1.0, 2.475, 3.95, 5.425, 6.9),
      3 -> Array(0.1, 0.7, 1.3, 1.9, 2.6))
    val fdtModel = FuzzyBinaryDecisionTree.train(traRdd, impurity, tNorm, maxDepth, maxBins,
      numClasses, categoricalFeaturesInfo, thresholdsFeatureInfo)

    FuzzyDecisionTreeSuite.validateClassifier(fdtModel, traArr.toSeq, 0.96)
    FuzzyDecisionTreeSuite.validateClassifier(fdtModel, tstArr.toSeq, 0.93)

  }

  // Test 3.5: Test FDT in real iris dataset
  test("Test 3.5: Multiclass classification with continuous features: Fuzzy Multi Decision Tree and product tNorm") {
    val (traArr, tstArr) = FuzzyDecisionTreeSuite.getIrisFromCsvToLabeledPoint
    traArr.length should be (135)
    tstArr.length should be (15)

    // Run Fuzzy Binary Decision Tree
    // We create uniform triangular fuzzy partitions (5 fuzzy sets for each feature)
    val traRdd = sc.parallelize(traArr)
    val impurity = "fuzzy_entropy"
    val tNorm = "product"
    val maxDepth = 5
    val maxBins = 100
    val numClasses = 3
    val categoricalFeaturesInfo = Map.empty[Int, Int]
    val thresholdsFeatureInfo = Map(
      0 -> Array(4.3, 5.2, 6.1, 7.0, 7.9),
      1 -> Array(2.0, 2.6, 3.2, 3.8, 4.4),
      2 -> Array(1.0, 2.475, 3.95, 5.425, 6.9),
      3 -> Array(0.1, 0.7, 1.3, 1.9, 2.6))
    val fdtModel = FuzzyMultiDecisionTree.train(traRdd, impurity, tNorm, maxDepth, maxBins,
      numClasses, categoricalFeaturesInfo, thresholdsFeatureInfo)

    FuzzyDecisionTreeSuite.validateClassifier(fdtModel, traArr.toSeq, 0.96)
    FuzzyDecisionTreeSuite.validateClassifier(fdtModel, tstArr.toSeq, 0.93)

  }

  // Test 3.6: Test FDT in real iris dataset
  test("Test 3.6: Multiclass classification with continuous features: Fuzzy Multi Decision Tree and minimum tNorm") {
    val (traArr, tstArr) = FuzzyDecisionTreeSuite.getIrisFromCsvToLabeledPoint
    traArr.length should be (135)
    tstArr.length should be (15)

    // Run Fuzzy Binary Decision Tree
    // We create uniform triangular fuzzy partitions (5 fuzzy sets for each feature)
    val traRdd = sc.parallelize(traArr)
    val impurity = "fuzzy_entropy"
    val tNorm = "min"
    val maxDepth = 5
    val maxBins = 100
    val numClasses = 3
    val categoricalFeaturesInfo = Map.empty[Int, Int]
    val thresholdsFeatureInfo = Map(
      0 -> Array(4.3, 5.2, 6.1, 7.0, 7.9),
      1 -> Array(2.0, 2.6, 3.2, 3.8, 4.4),
      2 -> Array(1.0, 2.475, 3.95, 5.425, 6.9),
      3 -> Array(0.1, 0.7, 1.3, 1.9, 2.6))
    val fdtModel = FuzzyMultiDecisionTree.train(traRdd, impurity, tNorm, maxDepth, maxBins,
      numClasses, categoricalFeaturesInfo, thresholdsFeatureInfo)

    FuzzyDecisionTreeSuite.validateClassifier(fdtModel, traArr.toSeq, 0.96)
    FuzzyDecisionTreeSuite.validateClassifier(fdtModel, tstArr.toSeq, 0.86)

  }

  // Test 3.7: Test FDT in real iris dataset
  test("Test 3.7: Multiclass classification with continuous features: FBDT, product tNorm and node cardinality (crisp and fuzzy)") {
    val (traArr, tstArr) = FuzzyDecisionTreeSuite.getIrisFromCsvToLabeledPoint
    traArr.length should be (135)
    tstArr.length should be (15)

    // Run Fuzzy Binary Decision Tree
    // We create uniform triangular fuzzy partitions (5 fuzzy sets for each feature)
    val traRdd = sc.parallelize(traArr)
    val impurity = "fuzzy_entropy"
    val tNorm = "product"
    val maxDepth = 5
    val maxBins = 100
    val numClasses = 3
    val categoricalFeaturesInfo = Map.empty[Int, Int]
    val thresholdsFeatureInfo = Map(
      0 -> Array(4.3, 5.2, 6.1, 7.0, 7.9),
      1 -> Array(2.0, 2.6, 3.2, 3.8, 4.4),
      2 -> Array(1.0, 2.475, 3.95, 5.425, 6.9),
      3 -> Array(0.1, 0.7, 1.3, 1.9, 2.6))
    val minInstancesPerNode = 80
    val minFuzzyInstancesPerNode = 50D
    val fdtModel = FuzzyBinaryDecisionTree.train(traRdd, impurity, tNorm, maxDepth, maxBins,
      numClasses, categoricalFeaturesInfo, thresholdsFeatureInfo, minInstancesPerNode,
      minFuzzyInstancesPerNode = minFuzzyInstancesPerNode)

    FuzzyDecisionTreeSuite.validateNodeCardinalities(fdtModel.topNode, minInstancesPerNode, minFuzzyInstancesPerNode)

  }

  // Test 3.8: Test FDT in real iris dataset
  test("Test 3.8: Multiclass classification with continuous features: FMDT, product tNorm and node cardinality (crisp and fuzzy)") {
    val (traArr, tstArr) = FuzzyDecisionTreeSuite.getIrisFromCsvToLabeledPoint
    traArr.length should be (135)
    tstArr.length should be (15)

    // Run Fuzzy Binary Decision Tree
    // We create uniform triangular fuzzy partitions (5 fuzzy sets for each feature)
    val traRdd = sc.parallelize(traArr)
    val impurity = "fuzzy_entropy"
    val tNorm = "product"
    val maxDepth = 5
    val maxBins = 100
    val numClasses = 3
    val categoricalFeaturesInfo = Map.empty[Int, Int]
    val thresholdsFeatureInfo = Map(
      0 -> Array(4.3, 5.2, 6.1, 7.0, 7.9),
      1 -> Array(2.0, 2.6, 3.2, 3.8, 4.4),
      2 -> Array(1.0, 2.475, 3.95, 5.425, 6.9),
      3 -> Array(0.1, 0.7, 1.3, 1.9, 2.6))
    val minInstancesPerNode = 50
    val minFuzzyInstancesPerNode = 50D
    val fdtModel = FuzzyBinaryDecisionTree.train(traRdd, impurity, tNorm, maxDepth, maxBins,
      numClasses, categoricalFeaturesInfo, thresholdsFeatureInfo, minInstancesPerNode,
      minFuzzyInstancesPerNode = minFuzzyInstancesPerNode)

    FuzzyDecisionTreeSuite.validateNodeCardinalities(fdtModel.topNode, minInstancesPerNode, minFuzzyInstancesPerNode)

  }

}

object FuzzyDecisionTreeSuite extends FunSuite with SharedSparkContext {

  def numLeaves(node: Node): Int = {
    var leaf = if (node.isLeaf) 1 else 0
    if (node.leftNode.nonEmpty){
      leaf += FuzzyDecisionTreeSuite.numLeaves(node.leftNode.get)
    }
    if (node.rightNode.nonEmpty){
      leaf += FuzzyDecisionTreeSuite.numLeaves(node.rightNode.get)
    }
    leaf
  }

  def validateClassifier(
                          model: FuzzyDecisionTreeModel,
                          input: Seq[LabeledPoint],
                          requiredAccuracy: Double) = {
    val predictions = input.map(x => model.predictByMaxValue(x.features))
    val numOffPredictions = predictions.zip(input).count { case (prediction, expected) =>
      prediction != expected.label
    }
    val accuracy = (input.length - numOffPredictions).toDouble / input.length
    assert(accuracy >= requiredAccuracy,
      s"validateClassifier calculated accuracy $accuracy but required $requiredAccuracy.")
  }

  def validateNodeCardinalities(node: iet.unipi.bigdatamining.classification.tree.model.Node,
                                minInstancesPerNode: Int, minFuzzyInstancesPerNode: Double) {
    if (node.numLeaves > 1){
      assert(node.cardinality >= minInstancesPerNode,
        s"Node validation error: number of instances at the node ${node.cardinality} " +
          s"but required $minInstancesPerNode.")
      assert(node.fuzzyCardinality >= minFuzzyInstancesPerNode,
        s"Node validation error: number of instances at the node ${node.fuzzyCardinality} " +
          s"but required $minFuzzyInstancesPerNode.")

      node.sons.foreach(node => validateNodeCardinalities(node, minInstancesPerNode, minFuzzyInstancesPerNode))
    }
  }

  def validatePredictions(
                           prediction1: Map[Int, Double],
                           prediction2: Map[Int, Double]
                         ) = {
    val df = new DecimalFormat("#.######")
    df.setRoundingMode(RoundingMode.HALF_UP)
    prediction1.mapValues(df.format(_).toDouble) should be (prediction2.mapValues(df.format(_).toDouble))

  }

  def generateOrderedLabeledPointsWithLabel0: Array[LabeledPoint] = {
    val arr = new Array[LabeledPoint](1000)
    for (i <- 0 until 1000) {
      val lp = new LabeledPoint(0D, Vectors.dense(i.toDouble, 1000D - i))
      arr(i) = lp
    }
    arr
  }

  def generateOrderedLabeledPointsWithLabel1: Array[LabeledPoint] = {
    val arr = new Array[LabeledPoint](1000)
    for (i <- 0 until 1000) {
      val lp = new LabeledPoint(1D, Vectors.dense(i.toDouble, 999D - i))
      arr(i) = lp
    }
    arr
  }

  def generateOrderedLabeledPoints: Array[LabeledPoint] = {
    val arr = new Array[LabeledPoint](1000)
    for (i <- 0 until 1000) {
      val label = if (i < 100) {
        0D
      } else if (i < 500) {
        1D
      } else if (i < 900) {
        0D
      } else {
        1D
      }
      arr(i) = new LabeledPoint(label, Vectors.dense(i.toDouble, 1000D - i))
    }
    arr
  }

  def generateCategoricalDataPoints: Array[LabeledPoint] = {
    val arr = new Array[LabeledPoint](1000)
    for (i <- 0 until 1000) {
      if (i < 600) {
        arr(i) = new LabeledPoint(1D, Vectors.dense(0D, 1D))
      } else {
        arr(i) = new LabeledPoint(0D, Vectors.dense(1D, 0D))
      }
    }
    arr
  }

  def generateCategoricalDataPointsForMulticlass: Array[LabeledPoint] = {
    val arr = new Array[LabeledPoint](3000)
    for (i <- 0 until 3000) {
      if (i < 1000) {
        arr(i) = new LabeledPoint(2D, Vectors.dense(2D, 2D))
      } else if (i < 2000) {
        arr(i) = new LabeledPoint(1D, Vectors.dense(1D, 2D))
      } else {
        arr(i) = new LabeledPoint(2D, Vectors.dense(2D, 2D))
      }
    }
    arr
  }

  def generateContinuousDataPointsForMulticlass: Array[LabeledPoint] = {
    val arr = new Array[LabeledPoint](3000)
    for (i <- 0 until 3000) {
      if (i < 2000) {
        arr(i) = new LabeledPoint(2D, Vectors.dense(2D, i))
      } else {
        arr(i) = new LabeledPoint(1D, Vectors.dense(2D, i))
      }
    }
    arr
  }

  def getIrisFromCsvToLabeledPoint: (Array[LabeledPoint], Array[LabeledPoint]) = {
    // Header: sepal-length, sepal-width, petal-length, petal-width, class
    val irisCSV = Source.fromURL(getClass.getResource("/iris.csv")).getLines()
    val (traSetCSV, tstSetCSV) = irisCSV.toArray.splitAt(135)
    // Training first 90% and Test last 10% of the overall data
    val trainingSet = traSetCSV.map{ row =>
      val point = row.split(",")
      val features = point.take(4).map(_.toDouble)
      val classLabel = point.last.toDouble
      new LabeledPoint(classLabel, Vectors.dense(features))
    }
    val testSet = tstSetCSV.map{ row =>
      val point = row.split(",")
      val features = point.take(4).map(_.toDouble)
      val classLabel = point.last.toDouble
      new LabeledPoint(classLabel, Vectors.dense(features))
    }

    (trainingSet, testSet)

  }

}