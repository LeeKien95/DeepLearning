package GR;


import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Random;


/**
 * Created by Welcome on 22/05/2017.
 */
public class MNISTDataClassificationVer2 {
//    public static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_Mnist/");

    private static Logger log = LoggerFactory.getLogger(MNISTDataClassificationVer2.class.getName());

    public static void main(String[] args) throws IOException {
//        log.warn("Hello World");
        int height = 28;
        int width = 28;
        int channel = 1;
        int rngSeed = 123;
        Random randNumGen = new Random(rngSeed);
        int batchSize = 128;
        int outputNum = 10;
        int numEpochs = 1;  // number of training times.

        File trainData = new File("D:/Programing/java/dl4j-examples/data/" + "/mnist_png/training");
        File testData = new File("D:/Programing/java/dl4j-examples/data/" + "/mnist_png/testing");
        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        ImageRecordReader recordReader = new ImageRecordReader(height, width, channel, labelMaker);

        recordReader.initialize(train);

        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);


        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);


        log.error("**********Build Model**********");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngSeed)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)
            .learningRate(0.006)
            .updater(Updater.NESTEROVS).momentum(0.9)   //momentum is for stop learningRate/weights from exploring or vanishing? or just keep weight descent in correct direction?
            .regularization(true).l2(1e-4)
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(height * width)
                .nOut(100)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .build())
            .layer(1, new DenseLayer.Builder()
                .nIn(1000)
                .nOut(800)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(800)
                .nOut(outputNum)
                .activation(Activation.SOFTMAX)
                .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .weightInit(WeightInit.XAVIER)
                .build())
            .pretrain(false).backprop(true)
            .setInputType(InputType.convolutional(height, width, chanel))
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);

        model.setListeners(new ScoreIterationListener(10));


        log.error("*****TRAIN MODEL********");
        for(int i = 0; i<numEpochs; i++){
            model.fit(dataIter);
        }

//        log.error("******EVALUATE MODEL******");
//
//        recordReader.reset();
//
//        recordReader.initialize(test);
//        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);
//        scaler.fit(testIter);
//        testIter.setPreProcessor(scaler);
//
//        log.error(recordReader.getLabels().toString());
//
//        // Create Eval object with 10 possible classes
//        Evaluation eval = new Evaluation(outputNum);
//
//        // Evaluate the network
//        while(testIter.hasNext()){
//            DataSet next = testIter.next();
//            INDArray output = model.output(next.getFeatureMatrix());
//            // Compare the Feature Matrix from the model
//            // with the labels from the RecordReader
//            eval.eval(next.getLabels(),output);
//
//        }
//        log.error(eval.stats());

        log.error("********Save model********");
        File locationToSave = new File("trained_mnist_model.zip");

        // boolean save Updater
        boolean saveUpdater = false;

        // ModelSerializer needs modelname, saveUpdater, Location

        ModelSerializer.writeModel(model,locationToSave,saveUpdater);

    }

}
