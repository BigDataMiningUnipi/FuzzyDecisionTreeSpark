# Fuzzy Decision Tree 

This project implements a *Fuzzy Decision Tree* (FDT) learning scheme built upon Apache Spark framework for generating FDTs from big data.

Publication:
A. Segatori, F. Marcelloni, W. Pedrycz, **"On Distributed Fuzzy Decision Trees for Big Data"**, *IEEE Transactions on Fuzzy Systems* 
DOI: <a href="https://doi.org/10.1109/TFUZZ.2016.2646746">10.1109/TFUZZ.2016.2646746</a>

## Main Features
Unlike classical Decision Trees (DTs), FDTs exploit fuzzy set theory to deal with uncertainty. In particular, each node is characterized by a fuzzy subset rather than a crisp set. Thus, each instance can activate different branches and reach multiple leaves. The integration between DTs and fuzzy set theory has proved to be very effective since it allows FDTs to be more robust to noise than DTs.

This code exploits the classical Decision Tree implementation in Spark Mllib, extending the learning scheme by employing fuzzy information gain based on fuzzy entropy and generating both *binary* and *multy-way* FDTs for *classification* problems. In particular, it assumes that a fuzzy partition is already defined on continuous features and then selects the best attribute of each node by employing the fuzzy entropy entropy. 

The algorithm has been tested by employing ten real-world publicly available big datasets, such as *Susy* and *Higgs*, evaluating the behavior of the scheme learning in terms of classification accuracy, model complexity, execution time, scalability (i.e. varying the number of cores and increasing dataset size) and comparing the results with the one achieved by Decision Tree implemented in Spark Mllib. 

For more details, please read PDF in the *doc* folder or download the original paper available at the following <a href="https://doi.org/10.1109/TFUZZ.2016.2646746">link</a>..

## Usage

### Packaging the application
Clone the repository and run the following command:
```sh
mvn clean package
```
Use in your own application the *unipi-fuzzy-decision-tree-1.0.jar* jar located in the *target* folder.
Check next section to see how to run the FDTs from your code.


### Examples
 
The examples below show how to run FDTs to perform classification using a fuzzy decision tree. Test error is calculated to measure the algorithm accuracy.

Input parameters are similar to <a href="http://spark.apache.org/docs/latest/mllib-decision-tree.html">Decision Tree</a> ones provided by Apache Mllib 

Configuration Parameters are:
- **impurity** (default value *fuzzy_entropy*): a *string* to store the impurity measure. The only accepted value is *fuzzy_entropy*
- **tNorm** (default value *Product*): a *string* to store the t-norm function used for the inference. Accepted values are *Product* and *Min*
- **maxDepth** (default value *10*): an *int* to store the maximum depth of Fuzzy Decision Tree 
- **maxBins** (default value *32*): an *int* to store the maximum number of bins for each feature 
- **numClasses** (default value *2*): an *int* to store the number of class labels. It can take values {*0*, ..., *numClasses - 1*}.
- **categoricalFeaturesInfo** (default value *Empty map*): a map that contains for each categorical feature (identified by its index) the number of categorical values. Similar to the labels if a categorical features contains *L* possible values, it can take values {*0*, ..., *L - 1*}
- **thresholdsFeatureInfo** (default value *Empty map*): a map that contains for each continuous feature (identified by its index) a list of cores of each triangular fuzzy set (FDTs will create a string triangular fuzzy partition from such lists)
- **minInstancesPerNode** (default value *1*): a *int* to store the minimum number of examples that each leaf must contain
- **minFuzzyInstancesPerNode** (default value *0*): a *double* to store the minimum node fuzzy cardinality to be inspected in the stop condition. Indeed, *node fuzzy cardinality* is computed as the sum of the membership degrees of all points in the dataset from the root to the node
- **minImpurityRatioPerNode** (default value *0*): a *double* to store the minimum ratio thresholds of impurity that must be true in each node
- **infoGain** (default value *0.00001*): a *double* to store the minimum information gain threshold that must be true in each node (If not true the node is not split).
- **subsamplingRate** (default value *1*): a double to store the ratio of subsampling (if 1 all dataset is considered)

#### Fuzzy Binary Decision Tree

Fuzzy Binary Decision Tree supports both Scala and Java. Here, how to train a Fuzzy Decision Tree model using the same inputs of the ones employed in the paper. 

##### Scala
```scala

import org.apache.spark.mllib.util.MLUtils

import iet.unipi.bigdatamining.classification.tree.FuzzyBinaryDecisionTree
import iet.unipi.bigdatamining.classification.tree.model.FuzzyDecisionTreeModel


// Load and parse the data file.
val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
// Split the data into training and test sets (20% held out for testing)
val splits = data.randomSplit(Array(0.8, 0.2))
val (trainingData, testData) = (splits(0), splits(1))

// Set parameters of Fuzzy Binary Decision Tree training (same of the ones employed in the paper).
//  Empty categoricalFeaturesInfo indicates all features are continuous.
//  Empty thresholdsFeatureInfo indicates all features are categorical

val impurity = "fuzzy_entropy"
val tNorm = "Product"
val maxDepth = 10
val maxBins = 32
val numClasses = 3
val categoricalFeaturesInfo = Map.empty[Int, Int]
val thresholdsFeatureInfo =  Map(
        0 -> Array(4.3, 5.2, 6.1, 7.0, 7.9), // Feature 0 -> 5 Fuzzy sets
        1 -> Array(2.0, 2.6, 3.2, 3.8, 4.4), // Feature 1 -> 5 Fuzzy sets
        2 -> Array(1.0, 2.475, 3.95, 6.9), // Feature 2 -> 4 Fuzzy sets
        3 -> Array(0.1, 1.9, 2.6)) // Feature 3 -> 3 Fuzzy sets

// Train a FuzzyBinaryDecisionTree model for classification.
val fdtModel = FuzzyBinaryDecisionTree.train(trainingData, 
        impurity, tNorm, maxDepth, maxBins, numClasses,
        categoricalFeaturesInfo, thresholdsFeatureInfo)


// Evaluate model on test instances and compute test error
val labelAndPreds = testData.map { point =>
  val prediction = model.predictByMaxValue(point.features)
  (point.label, prediction)
}
val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()

// Print model complexity
println(s"#Node: ${fdtModel.numNodes}")
println(s"#Leaves: ${fdtModel.numLeaves}")
println(s"#MaxDepth: ${fdtModel.depth}")
println(s"#MinDepth: ${fdtModel.minDepth}")
println(s"#AvgDepth: ${fdtModel.averageDepth}")

// Print accuracy and model
println(s"Test Error = $testErr")
println(s"Learned classification tree model:\n${fdtModel.toDebugString}") // Set maxDepth=5 for a more readable model

``` 

##### Java
```java
import java.util.HashMap;
import java.util.Map;

import scala.Tuple2;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

import iet.unipi.bigdatamining.classification.tree.FuzzyBinaryDecisionTree
import iet.unipi.bigdatamining.classification.tree.model.FuzzyDecisionTreeModel

SparkConf sparkConf = new SparkConf().setAppName("JavaFuzzyBinaryDecisionTreeExample");
JavaSparkContext jsc = new JavaSparkContext(sparkConf);

// Load and parse the data file.
String datapath = "data/mllib/sample_libsvm_data.txt";
JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), datapath).toJavaRDD();
// Split the data into training and test sets (30% held out for testing)
JavaRDD<LabeledPoint>[] splits = data.randomSpli2(new double[]{0.8, 0.2});
JavaRDD<LabeledPoint> trainingData = splits[0];
JavaRDD<LabeledPoint> testData = splits[1];

// Set parameters of Fuzzy Binary Decision Tree training (same of the ones employed in the paper).
//  Empty categoricalFeaturesInfo indicates all features are continuous.
//  Empty thresholdsFeatureInfo indicates all features are categorical

String impurity = "fuzzy_entropy";
String tNorm = "Product";
Integer maxDepth = 10;
Integer maxBins = 32;
Integer numClasses = 3;
Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
Map<Integer, List<Double>> thresholdsFeatureInfo =  new HashMap<Integer, List<Double>>();
thresholdsFeatureInfo.put(0, Arrays.asList(4.3, 5.2, 6.1, 7.0, 7.9)); // Feature 0 -> 5 Fuzzy sets
thresholdsFeatureInfo.put(1, Arrays.asList(2.0, 2.6, 3.2, 3.8, 4.4)); // Feature 1 -> 5 Fuzzy sets
thresholdsFeatureInfo.put(2, Arrays.asList(1.0, 2.475, 3.95, 6.9)); // Feature 2 -> 4 Fuzzy sets
thresholdsFeatureInfo.put(3, Arrays.asList(0.1, 1.9, 2.6)); // Feature 3 -> 3 Fuzzy sets
Integer minInstancesPerNode = 1;
Double minFuzzyInstancesPerNode = 0D;
Double minImpurityRatioPerNode = 1D;
Double minInfoGain = 0.000001;
Double subsamplingFraction = 1D;

// Train a FuzzyBinaryDecisionTree model.
final FuzzyDecisionTreeModel fdtModel = FuzzyBinaryDecisionTree.trainFromJava(trainingData,
        impurity, tNorm, maxDepth, maxBins, numClasses,
		categoricalFeaturesInfo, thresholdsFeatureInfo, minInstancesPerNode,
		minFuzzyInstancesPerNode, minImpurityRatioPerNode, minInfoGain, subsamplingFraction);

// Evaluate model on test instances and compute test error
JavaPairRDD<Double, Double> predictionAndLabel =
  testData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
    @Override
    public Tuple2<Double, Double> call(LabeledPoint p) {
      return new Tuple2<Double, Double>(model.predictByMaxValue(p.features()), p.label());
    }
  });
Double testErr =
  1D * predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
    @Override
    public Boolean call(Tuple2<Double, Double> pl) {
      return !pl._1().equals(pl._2());
    }
  }).count() / testData.count();

// Print model complexity
System.out.println("#Node: " + fdtModel.numNodes());
System.out.println("#Leaves: " + fdtModel.numLeaves());
System.out.println("#MaxDepth: " + fdtModel.depth());
System.out.println("#MinDepth: " + fdtModel.minDepth());
System.out.println("AvgDepth: " + fdtModel.averageDepth());

// Print accuracy and model 
System.out.println("Test Error: " + testErr);
System.out.println("Learned classification tree model:\n" + fdtModel.toDebugString());

``` 


#### Fuzzy Multi Decision Tree

Fuzzy Multi Decision Tree supports both Scala and Java. Here, how to train a Fuzzy Decision Tree model using the same inputs of the ones employed in the paper. 

##### Scala
```scala

import org.apache.spark.mllib.util.MLUtils

import iet.unipi.bigdatamining.classification.tree.FuzzyMultiDecisionTree
import iet.unipi.bigdatamining.classification.tree.model.FuzzyDecisionTreeModel


// Load and parse the data file.
val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
// Split the data into training and test sets (20% held out for testing)
val splits = data.randomSplit(Array(0.8, 0.2))
val (trainingData, testData) = (splits(0), splits(1))

// Set parameters of Fuzzy Multi Decision Tree training (same of the ones employed in the paper).
//  Empty categoricalFeaturesInfo indicates all features are continuous.
//  Empty thresholdsFeatureInfo indicates all features are categorical

val impurity = "fuzzy_entropy"
val tNorm = "Product"
val maxDepth = 5
val maxBins = 32
val numClasses = 3
val categoricalFeaturesInfo = Map(
        0 -> 5, // Feature 0 has 5 different values (values should be 0-based, namely {0,1,2,3,4})
        1 -> 2, // Feature 1 has 2 different values (values should be 0-based, namely {0,1})
        2 -> 7) // Feature 2 has 7 different values (values should be 0-based, namely {0,1,2})
val thresholdsFeatureInfo =  Map.empty[Int, List[Double]]
val minInstancesPerNode = trainingData.count / 10000; (0.01% of the total number of instances)

// Train a FuzzyMultiDecisionTree model for classification.
val fdtModel = FuzzyMultiDecisionTree.train(trainingData, 
        impurity, tNorm, maxDepth, maxBins, numClasses,
        categoricalFeaturesInfo, thresholdsFeatureInfo,
        minInstancesPerNode)

// Evaluate model on test instances and compute test error
val labelAndPreds = testData.map { point =>
  val prediction = model.predictByMaxValue(point.features)
  (point.label, prediction)
}
val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()

// Print model complexity
println(s"#Node: ${fdtModel.numNodes}")
println(s"#Leaves: ${fdtModel.numLeaves}")
println(s"#MaxDepth: ${fdtModel.depth}")
println(s"#MinDepth: ${fdtModel.minDepth}")
println(s"#AvgDepth: ${fdtModel.averageDepth}")

// Print accuracy and model
println(s"Test Error = $testErr")
println(s"Learned classification tree model:\n${fdtModel.toDebugString}") // Set maxDepth=5 for a more readable model

``` 

##### Java
```java
import java.util.HashMap;
import java.util.Map;

import scala.Tuple2;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

import iet.unipi.bigdatamining.classification.tree.FuzzyMultiDecisionTree
import iet.unipi.bigdatamining.classification.tree.model.FuzzyDecisionTreeModel

SparkConf sparkConf = new SparkConf().setAppName("JavaFuzzyBinaryDecisionTreeExample");
JavaSparkContext jsc = new JavaSparkContext(sparkConf);

// Load and parse the data file.
String datapath = "data/mllib/sample_libsvm_data.txt";
JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), datapath).toJavaRDD();
// Split the data into training and test sets (30% held out for testing)
JavaRDD<LabeledPoint>[] splits = data.randomSpli2(new double[]{0.8, 0.2});
JavaRDD<LabeledPoint> trainingData = splits[0];
JavaRDD<LabeledPoint> testData = splits[1];

// Set parameters of Fuzzy Multi Decision Tree training (same of the ones employed in the paper).
//  Empty categoricalFeaturesInfo indicates all features are continuous.
//  Empty thresholdsFeatureInfo indicates all features are categorical

String impurity = "fuzzy_entropy";
String tNorm = "Product";
Integer maxDepth = 5;
Integer maxBins = 32;
Integer numClasses = 3;
Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
categoricalFeaturesInfo.put(0, 5); // Feature 0 has 5 different values (values should be 0-based, namely {0,1,2,3,4})
categoricalFeaturesInfo.put(1, 2); // Feature 0 has 2 different values (values should be 0-based, namely {0,1}) 
categoricalFeaturesInfo.put(2, 7); // Feature 0 has 7 different values (values should be 0-based, namely {0,1,2}) 
Map<Integer, List<Double>> thresholdsFeatureInfo =  new HashMap<Integer, List<Double>>();
Integer minInstancesPerNode = trainingData.count() / 10000; (0.01% of the total number of instances)
Double minFuzzyInstancesPerNode = 0D;
Double minImpurityRatioPerNode = 1D;
Double minInfoGain = 0.000001;
Double subsamplingFraction = 1D;

// Train a FuzzyBinaryDecisionTree model.
final FuzzyDecisionTreeModel fdtModel = FuzzyMultiDecisionTree.trainFromJava(trainingData,
        impurity, tNorm, maxDepth, maxBins, numClasses,
		categoricalFeaturesInfo, thresholdsFeatureInfo, minInstancesPerNode,
		minFuzzyInstancesPerNode, minImpurityRatioPerNode, minInfoGain, subsamplingFraction);

// Evaluate model on test instances and compute test error
JavaPairRDD<Double, Double> predictionAndLabel =
  testData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
    @Override
    public Tuple2<Double, Double> call(LabeledPoint p) {
      return new Tuple2<Double, Double>(model.predictByMaxValue(p.features()), p.label());
    }
  });
Double testErr =
  1D * predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
    @Override
    public Boolean call(Tuple2<Double, Double> pl) {
      return !pl._1().equals(pl._2());
    }
  }).count() / testData.count();

// Print model complexity
System.out.println("#Node: " + fdtModel.numNodes());
System.out.println("#Leaves: " + fdtModel.numLeaves());
System.out.println("#MaxDepth: " + fdtModel.depth());
System.out.println("#MinDepth: " + fdtModel.minDepth());
System.out.println("AvgDepth: " + fdtModel.averageDepth());

// Print accuracy and model 
System.out.println("Test Error: " + testErr);
System.out.println("Learned classification tree model:\n" + fdtModel.toDebugString());

``` 

#### Important Notes

As described in the previous snippets, FDTs scheme learning takes fuzzy partitions as input, therefore you have to define different fuzzy partitions for each continuous feature.

To run the same algorithm detailed in the original paper, please use the Fuzzy Partitioning algorithm available <a href="https://github.com/BigDataMiningUnipi/FuzzyPartitioningSpark">here</a>

Here an example of the overall code (Fuzzy Partitioning + Fuzzy Binary Decision Tree):
```scala



// Load and parse the data file.
val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
// Split the data into training and test sets (20% held out for testing)
val splits = data.randomSplit(Array(0.8, 0.2))
val (trainingData, testData) = (splits(0), splits(1))

//////////////////////////////////
//      Fuzzy Partitioning      
//////////////////////////////////
// Set parameters of Fuzzy Partitioing (same of the ones employed in the paper).
//  Empty categoricalFeaturesInfo indicates all features are continuous.
val numFeatures = 4
val numClasses = 3
val categoricalFeatures = Set(1,2)
val impurity = "FuzzyEntropy"
val maxBins = 1000
val candidateSplitStrategy = "EquiFreqPerPartition"
val minInfoGain = 0.000001
val subsamplingFraction = 1D
val minInstancesPerSubsetRatio = 0.0001D

// Run Fuzzy Partitioinng
val fpModel = FuzzyPartitioning.discretize(trainingData, numFeatures, numClasses, categoricalFeatures, impurity,
      maxBins, candidateSplitStrategy, minInfoGain, subsamplingFraction, minInstancesPerSubsetRatio)


// Print model complexity
println(s"Totoal number of Fuzzy Sets: ${fpModel.numFuzzySets}") 
println(s"Average number of Fuzzy Sets: ${fpModel.averageFuzzySets}") 
println(s"Number of discarded features: ${fpModel.discardedFeature}") 
println(s"Number of fuzzy sets of the feature with the highest number of fuzzy sets: ${fpModel.max._2}") 
println(s"Number of fuzzy sets if the feature with the lowest number of fuzzy sets (discarded features are not taken in considiration): ${fpModel.min._2}")


//////////////////////////////////
//     Fuzzy Decision Tree      
//////////////////////////////////
// Remove features with 0 cut points
val cores = fpModel.cores 
val featuresToRemove = fpModel.discardedFeature
val trainingDataFiltered = ??? /*Remove those features that are in featuresToRemove */
val testDataFiltered = ??? /*Remove those features that are in featuresToRemove */
val thresholdsFeatureInfo =  ??? /*Remove from cores discarded features and update the feature index to keep them aligned with trainingDataFiltered and testDataFiltered*/


// Run the FDT algorihtm
//   Empty categoricalFeaturesInfo indicates all features are continuous.
//   Empty thresholdsFeatureInfo indicates all features are categorical.

val impurity = "fuzzy_entropy"
val tNorm = "Product"
val maxDepth = 10
val maxBins = 32
val numClasses = 3
val categoricalFeaturesInfo = Map.empty[Int, Int]

// Train a FuzzyBinaryDecisionTree model for classification.
val fdtModel = FuzzyBinaryDecisionTree.train(trainingDataFiltered, 
        fdtImpurity, fdtTNorm, fdtMaxDepth, currentMaxBins, numClasses,
        categoricalFeaturesInfo, thresholdsFeatureInfo)


// Evaluate model on test instances and compute test error
val labelAndPreds = testDataFiltered.map { point =>
  val prediction = model.predictByMaxValue(point.features)
  (point.label, prediction)
}
val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()

// Print FDT Model complexity
println(s"#Node: ${fdtModel.numNodes}")
println(s"#Leaves: ${fdtModel.numLeaves}")
println(s"#MaxDepth: ${fdtModel.depth}")
println(s"#MinDepth: ${fdtModel.minDepth}")
println(s"#AvgDepth: ${fdtModel.averageDepth}")

// Print accuracy and model
println(s"Test Error = $testErr")
println(s"Learned classification tree model:\n${fdtModel.toDebugString}") // Set maxDepth=5 for a more readable model

``` 

## Unsupported Features
- Sparse Vectors
- Saving/Loading of Models
- ML package


## Contributors

- <a href="https://it.linkedin.com/in/armandosegatori">Armando Segatori</a> (main contributor and maintainer)
- <a href="http://www.iet.unipi.it/f.marcelloni/">Francesco Marcelloni</a>
- <a href="http://www.ece.ualberta.ca/~pedrycz/">Witold Pedrycz</a>


## References

[1] A. Segatori, F. Marcelloni, W. Pedrycz, **"On Distributed Fuzzy Decision Trees for Big Data"**, *IEEE Transactions on Fuzzy Systems*

**Please cite the above work in your manuscript if you plan to use this code**:

###### Plain Text:
```
A. Segatori; F. Marcelloni; W. Pedrycz, "On Distributed Fuzzy Decision Trees for Big Data," in IEEE Transactions on Fuzzy Systems , vol.PP, no.99, pp.1-1
doi: 10.1109/TFUZZ.2016.2646746
```

###### BibTeX
```
@ARTICLE{7803561, 
    author={A. Segatori and F. Marcelloni and W. Pedrycz}, 
    journal={IEEE Transactions on Fuzzy Systems}, 
    title={On Distributed Fuzzy Decision Trees for Big Data}, 
    year={2017}, 
    volume={PP}, 
    number={99}, 
    pages={1-1}, 
    doi={10.1109/TFUZZ.2016.2646746}, 
    ISSN={1063-6706}, 
    month={},}
```


Are you looking for a way for generating *Fuzzy Partitions* for Big Data? Check out the **Fuzzy Partitioning** code available <a href="https://github.com/BigDataMiningUnipi/FuzzyPartitioningSpark">here</a>
