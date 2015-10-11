import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator

import org.deeplearning4j.eval.Evaluation

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{Updater, MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.layers.{OutputLayer, RBM}

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.params.DefaultParamInitializer
import org.deeplearning4j.nn.weights.WeightInit

import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.{DataSet, SplitTestAndTrain}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

import org.slf4j.LoggerFactory
import java.util.{Arrays, Random}

object Iris extends App {

  lazy val log = LoggerFactory.getLogger(Iris.getClass)

  val numRows = 4
  val numColumns = 1
  val outputNum = 3
  val numSamples = 150
  val batchSize = 150

  // Magic numbers
  val seed = 123
  val iterations = 5
  val splitTrainNum = (batchSize * .8).toInt


  log.info("Load data....")
  // https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/resources/iris.dat
  /*
  https://archive.ics.uci.edu/ml/datasets/Iris
  https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
  1. sepal length in cm
  2. sepal width in cm
  3. petal length in cm
  4. petal width in cm
  5. class:
  -- Iris Setosa        0
  -- Iris Versicolour   1
  -- Iris Virginica     2
  https://en.wikipedia.org/wiki/Iris_flower_data_set
*/

  val iter: DataSetIterator = new IrisDataSetIterator(batchSize, numSamples)
  val next: DataSet = iter.next()
  next.normalizeZeroMeanZeroUnitVariance()

  log.info("Split data....")
  val testAndTrain: SplitTestAndTrain = next.splitTestAndTrain(splitTrainNum, new Random(seed))
  val train = testAndTrain.getTrain()
  val test = testAndTrain.getTest()
  Nd4j.ENFORCE_NUMERICAL_STABILITY = true

  log.info("Build model....")

  val hiddenLayer = new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
  .nIn(numRows * numColumns) // # input nodes
  .nOut(3) // # fully connected hidden layer nodes. Add list if multiple layers.
  .weightInit(WeightInit.XAVIER) // Weight initialization
  .k(1) // # contrastive divergence iterations
  .activation("relu") // Activation function type
  .lossFunction(LossFunctions.LossFunction.RMSE_XENT) // Loss function type
  .updater(Updater.ADAGRAD)
  .dropOut(0.5)
  .build()

  val outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
  .nIn(3) // # input nodes
  .nOut(outputNum) // # output nodes
  .activation("softmax")
  .build()

  val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
  .seed(seed) // Locks in weight initialization for tuning
  .iterations(iterations) // # training iterations predict/classify & backprop
  .learningRate(1e-6f) // Optimization step size
  .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT) // Backprop to calculate gradients
  .l1(1e-1).regularization(true).l2(2e-4)
  .useDropConnect(true)
  .list(2) // # NN layers (doesn't count input layer)
  .layer(0, hiddenLayer)
  .layer(1, outputLayer)
  .build()

  val model: MultiLayerNetwork = new MultiLayerNetwork(conf)
  model.init()
  //  model.setListeners(Arrays.asList(new ScoreIterationListener(listenerFreq),
  //          new GradientPlotterIterationListener(listenerFreq),
  //          new LossPlotterIterationListener(listenerFreq)))

  val listenerFreq = 1
  val listener: IterationListener = new ScoreIterationListener(listenerFreq)
  model.setListeners(Arrays.asList(listener))
  log.info("Train model....")
  model.fit(train)

  // log.info("Evaluate weights....")
  // for(org.deeplearning4j.nn.api.Layer layer : model.getLayers()) {
  //     INDArray w = layer.getParam(DefaultParamInitializer.WEIGHT_KEY)
  //     log.info("Weights: " + w)
  // }

  log.info("Evaluate model....")
  val eval: Evaluation = new Evaluation(outputNum)
  val output: INDArray = model.output(test.getFeatureMatrix())

  (0 until output.rows).foreach { i =>
    val actual = test.getLabels().getRow(i).toString().trim()
    val predicted = output.getRow(i).toString().trim()
    log.info("actual " + actual + " vs predicted " + predicted)
  }

  eval.eval(test.getLabels(), output)
  log.info(eval.stats())
}