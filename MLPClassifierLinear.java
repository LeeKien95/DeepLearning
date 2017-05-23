package GR;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

import static org.deeplearning4j.nn.conf.layers.DenseLayer.*;

/**
 * Created by Welcome on 25/04/2017.
 */
public class MLPClassifierLinear {
    public static void main(String[] args) throws Exception{
        int seed = 123;
        double learningrate = 0.01;
        int batchSize = 50;
        int nEpochs = 30;
        int numInput = 2;
        int numOutput = 2;
        int numHiddenNodes = 20;

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new ClassPathResource("/classification/linear_data_train.csv").getFile()));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0,2);


        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new ClassPathResource("/classification/linear_data_eval.csv").getFile()));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0,2);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(1)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(learningrate)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .list()
            .layer(0, new DenseLayer.Builder()
                    .nIn(numInput)
                    .nOut(numHiddenNodes)
                    .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .weightInit(WeightInit.XAVIER)
                .activation("softmax")
                .weightInit(WeightInit.XAVIER)
                .nIn(numHiddenNodes)
                .nOut(numOutput)
                .build()
            )
            .pretrain(false)
            .backprop(true)
            .build();
        System.out.println(conf.toJson());
    }
}
