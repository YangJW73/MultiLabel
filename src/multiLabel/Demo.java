package multiLabel;

import mulan.classifier.MultiLabelOutput;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
//import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class Demo {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub

		try {
			//MultiLabelInstances train = new MultiLabelInstances("/Users/YangJW/Codes/workspace/MultiReinforcement/emotions/emotions-train.arff", "/Users/YangJW/Codes/workspace/MultiReinforcement/emotions/emotions.xml");
			//MultiLabelInstances test = new MultiLabelInstances("/Users/YangJW/Codes/workspace/MultiReinforcement/emotions/emotions-test.arff", "/Users/YangJW/Codes/workspace/MultiReinforcement/emotions/emotions.xml");

			MultiLabelInstances train = new MultiLabelInstances("/Users/YangJW/Codes/workspace/MultiReinforcement/scene/scene-train.arff", "/Users/YangJW/Codes/workspace/MultiReinforcement/scene/scene.xml");
			MultiLabelInstances test = new MultiLabelInstances("/Users/YangJW/Codes/workspace/MultiReinforcement/scene/scene-test.arff", "/Users/YangJW/Codes/workspace/MultiReinforcement/scene/scene.xml");
	        //Classifier brClassifier = new NaiveBayes();
	        /*MLPolicyBoostOL MLPB = new MLPolicyBoostOL();
	        MLPB.setDebug(true);
	        MLPB.build(train);*/
	        MLPolicyBoost MLPB = new MLPolicyBoost(1,5,6,false,0);
	        MLPB.build(train);
	        //MLPB.buildInternal(train);
	        System.out.println("model built!");
	        
	        Instances testdata = test.getDataSet();
	        int []testLabelIndices = test.getLabelIndices();
	        double hammingLoss = 0;
	        for(int i = 0; i<testdata.numInstances(); i++){
	        	Instance inst1 = new Instance(testdata.instance(i));
	        	Instance inst2 = new Instance(testdata.instance(i));
	        	MultiLabelOutput output = MLPB.makePrediction(inst1);
	        	boolean[] predictResult = output.getBipartition();
	        	int symmetricDifference = 0;
	        	for(int j = 0; j<testLabelIndices.length; j++){
	        		if(predictResult[j] && inst2.value(testLabelIndices[j])!=1){
	        			symmetricDifference++;
	        		}
	        		else if(!predictResult[j] && inst2.value(testLabelIndices[j])!=0){
	        			symmetricDifference++;
	        		}
	        	}
	        	hammingLoss= hammingLoss+(double)symmetricDifference/predictResult.length;
	        	//System.out.println(output.toString());
	        }
	        double result = (double) hammingLoss/testdata.numInstances();
	        System.out.println("Hamming loss :"+result);
	        /*Evaluator eval = new Evaluator();
            Evaluation results;
	        results = eval.evaluate(MLPB, test);
	        System.out.println(results);*/
	        
		} catch (InvalidDataFormatException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        

        
	}

}
