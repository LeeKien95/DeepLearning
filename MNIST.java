package GR;


import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * Created by Welcome on 13/05/2017.
 */

public class MNIST {
    private static Logger log = LoggerFactory.getLogger(MNIST.class);

    public static void main(String[] args) throws IOException {
        final int numRows = 28;     // number of rows of matrix
        final int numColumns = 28;  // number of columns of matrix.
        int outputNum = 10;         // number of possible output (0 to 9)
        int batchSize = 128;        // number of examples to fetch with each step
        int rngSeed = 123;          // ?
        int numEpochs = 15;         // ?


        //fetching the MNIST Data
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);
        /*
        * this class called DatasetIterator is used to fetch the MNIST dataset, with params declared before (batchSize, rngSeed)
        * mnistTrain is for training, mnistTest is for testing (2nd param = true/false)*/

        /*we will construct a shallow neural network with 1 hidden layer*/

        /*Setting Hyperparameters*/
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngSeed) //include a random seed for reproducibility
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)
            .learningRate(0.01)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .regularization(true).l2(1e-4)
            .list()
        /*Building Layers*/
            .layer(0, new DenseLayer.Builder() // input layer
                .nIn(numRows * numColumns)
                .nOut(1000)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .build())
            .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(1000)
                .nOut(outputNum)
                .activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER)
                .build())
            .pretrain(false).backprop(true) //use backpropagation to adjust weights
            .build();
        /*
        *For any neural network build by DL4J, the foundation is the NeuralNetConfiguration Class.
        * This is where you configure hyperparameters, the quantities that define the architecture and how the algorithm learns.
        *
        * seed(rngSeed) : this parameter is uses to keep initialized weight unchange for each time we run example again
        * optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT): choose Algorithm to learn: SDG
        * iterations(1): each iterations is a step aka an update of the model's weights.
        * updater(Updater.NESTEROVS).momentum(0.9): specify the rate of change of the learning rate.
        * learningRate(0.006): set the learning rate
        * regularization(true).l2(1e-4): technique to prevent overfitting. L2 regularization help to prevents invidiual weights from having too much influence on overall results.
        * list() : the list specifies the munber of layers in the net.*/

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        //print the score with every 1 iteration
        model.setListeners(new ScoreIterationListener(1));

        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        //Then add the StatsListener to collect this information from the network, as it trains
        model.setListeners(new StatsListener(statsStorage));

        log.info("Train model....");
        for( int i=0; i<numEpochs; i++ ){
            model.fit(mnistTrain);
        }


        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
        while(mnistTest.hasNext()){
            DataSet next = mnistTest.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }

        log.info(eval.stats());
        log.info("****************Example finished********************");

    }
}
