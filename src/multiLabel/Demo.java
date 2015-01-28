package multiLabel;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;

import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.EnsembleOfClassifierChains;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.ClassifierChain;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import weka.classifiers.trees.J48;
//import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class Demo {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub

		try {
			MultiLabelInstances dataSet = new MultiLabelInstances(".\\data\\emotions\\emotions.arff", ".\\data\\emotions\\emotions.xml");
			//MultiLabelInstances dataSet = new MultiLabelInstances(".\\data\\yeast\\yeast.arff", ".\\data\\yeast\\yeast.xml");
			//MultiLabelInstances dataSet = new MultiLabelInstances(".\\data\\scene\\scene.arff", ".\\data\\scene\\scene.xml");
			//MultiLabelInstances dataSet = new MultiLabelInstances(".\\data\\enron\\enron.arff", ".\\data\\enron\\enron.xml");
			//MultiLabelInstances dataSet = new MultiLabelInstances(".\\data\\medical\\medical.arff", ".\\data\\medical\\medical.xml");
			//MultiLabelInstances dataSet = new MultiLabelInstances(".\\data\\genbase\\genbase.arff", ".\\data\\genbase\\genbase.xml");
			//MultiLabelInstances dataSet = new MultiLabelInstances(".\\data\\bibtex\\bibtex.arff", ".\\data\\bibtex\\bibtex.xml");
	       
	        MLPolicyBoost MLPB = new MLPolicyBoost(5,30,false,0);
			//ClassifierChain MLPB = new ClassifierChain();
			//EnsembleOfClassifierChains MLPB = new EnsembleOfClassifierChains();
			//BinaryRelevance MLPB = new BinaryRelevance(new J48());
			
	        MyEvaluation eval = new MyEvaluation();
	        long start = System.currentTimeMillis();
	        eval.crossValidation(MLPB, 10, dataSet);
	        long end = System.currentTimeMillis();
	        System.out.println("running time: "+(end-start));
	        //eval.crossValidation(ecc, 10, dataSet);
		} catch (InvalidDataFormatException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        

        
	}

}
