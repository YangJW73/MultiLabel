package multiLabel;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;

import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.EnsembleOfClassifierChains;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
//import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class Demo {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub

		try {
			//MultiLabelInstances dataSet = new MultiLabelInstances(".\\data\\emotions\\emotions.arff", ".\\data\\emotions\\emotions.xml");
			//MultiLabelInstances test = new MultiLabelInstances(".\\data\\emotions\\emotions-test.arff", ".\\data\\emotions\\emotions.xml");

			MultiLabelInstances dataSet = new MultiLabelInstances(".\\data\\scene\\scene-train.arff", ".\\data\\scene\\scene.xml");
			//MultiLabelInstances test = new MultiLabelInstances(".\\data\\scene\\scene-test.arff", ".\\data\\scene\\scene.xml");
	       
	        MLPolicyBoost MLPB = new MLPolicyBoost(5,5,6,false,0);
	        //EnsembleOfClassifierChains ecc = new EnsembleOfClassifierChains();
	        System.out.println("model built!");
	        Evaluation eval = new Evaluation();
	        eval.crossValidation(MLPB, 10, dataSet);
	        //eval.crossValidation(ecc, 10, dataSet);
		} catch (InvalidDataFormatException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        

        
	}

}
