package GR;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * Created by Admin on 5/24/2017.
 */
public class MNISTImageClassificationModelLoadAndDemo {

    private static Logger log = LoggerFactory.getLogger(MNISTImageClassificationModelLoadAndDemo.class);

    public static String fileChose(){
        JFileChooser fc = new JFileChooser();
        int ret = fc.showOpenDialog(null);
        if (ret == JFileChooser.APPROVE_OPTION)
        {
            File file = fc.getSelectedFile();
            String filename = file.getAbsolutePath();
            return filename;
        }
        else {
            return null;
        }
    }

    public static void main(String[] args) throws IOException {
        int height = 28;
        int width = 28;
        int channels = 1;

        List<Integer> labelList = Arrays.asList(0,1,2,3,4,5,6,7,8,9);
        // pop up file chooser
        String filechose = fileChose().toString();

        //LOAD NEURAL NETWORK

        // Where to save model
        File locationToSave = new File("trained_mnist_model.zip");
        // Check for presence of saved model
        if(locationToSave.exists()){
            System.out.println("\n######Saved Model Found######\n");
        }else{
            System.out.println("\n\n#######File not found!#######");
            System.out.println("This example depends on running ");
            System.out.println("MnistImagePipelineExampleSave");
            System.out.println("Run that Example First");
            System.out.println("#############################\n\n");


            System.exit(0);
        }

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

        log.error("*********TEST YOUR IMAGE AGAINST SAVED NETWORK********");

        // FileChose is a string we will need a file

        File file = new File(filechose);

        // Use NativeImageLoader to convert to numerical matrix

        NativeImageLoader loader = new NativeImageLoader(height, width, channels);

        // Get the image into an INDarray

        INDArray image = loader.asMatrix(file);

        // 0-255
        // 0-1
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.transform(image);
        // Pass through to neural Net

        INDArray output = model.output(image);

        log.error("## The FILE CHOSEN WAS " + filechose);
        log.error("## The Neural Nets Pediction ##");
        log.error("## list of probabilities per label ##");
        //log.info("## List of Labels in Order## ");
        // In new versions labels are always in order
        log.error(output.toString());
        log.error(labelList.toString());
    }
}
