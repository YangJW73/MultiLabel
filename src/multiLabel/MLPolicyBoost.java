package multiLabel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.TechnicalInformation;
import mulan.classifier.*;
import mulan.classifier.transformation.ClassifierChain;
import mulan.data.DataUtils;
import mulan.data.MultiLabelInstances;
import domain.*;
import core.*;


public class MLPolicyBoost extends MultiLabelLearnerBase{
	
	PolicyBoostDiscrete policy;
	MLTask task;
	int iteration;
	int trialsPerIter;
	int maxStep;
	boolean isPara;
	double epsion;
	ArrayList<FilteredClassifier> classifierChain;
	ClassifierChain ensemble;
	ArrayList<Integer> actionList;
	int labelIndices[];
	MultiLabelInstances dataSet;
	
	public MLPolicyBoost(int iteration, int trialsPerIter, boolean isPara, double epsion){
		this.iteration = iteration;
		this.trialsPerIter = trialsPerIter;
		
		this.isPara = isPara;
		this.epsion = epsion;
	}

	@Override
	protected void buildInternal(MultiLabelInstances trainingSet)
			throws Exception {
		// TODO Auto-generated method stub
		Random random = new Random();
		numLabels = trainingSet.getNumLabels();
		this.maxStep = numLabels;
		dataSet = trainingSet;
		policy = new PolicyBoostDiscrete (new Random(random.nextInt()), Integer.MAX_VALUE);
		task = new MLTask(trainingSet, random);
		classifierChain = new ArrayList<FilteredClassifier>();
		actionList = new ArrayList<Integer>();
		labelIndices = trainingSet.getLabelIndices();
		//MLState initialState = (MLState) task.getInitialState();
		
		MLExperiment mle = new MLExperiment();
		mle.trainPolicy(policy, task, iteration, trialsPerIter/*, initialState*/, maxStep, isPara, epsion, new Random(random.nextInt()));
		int[] chain = new int[numLabels];
		for(int i = 0; i<mle.bestTraj.getSamples().size(); i++){
			MLState tmpS = (MLState)mle.bestTraj.getSamples().get(i).sPrime;
			Action tmpA = mle.bestTraj.getSamples().get(i).action;
			classifierChain.add(tmpS.c);
			actionList.add(tmpA.a);
			//chain[i] = mle.bestTraj.getSamples().get(i).action.a;
		}
		/*ensemble = new ClassifierChain(new J48(), chain);
        ensemble.build(dataSet);*/
		System.out.println("model built!");
	}

	@Override
	protected MultiLabelOutput makePredictionInternal(Instance instance)
			throws Exception, InvalidDataException {
		// TODO Auto-generated method stub
		/*boolean[] predictions = new boolean[numLabels];
		int cnt = 0;
		for(Classifier c : classifierChain){
			dataSet.getDataSet().setClassIndex(labelIndices[actionList.get(cnt)]);
			instance.setDataset(dataSet.getDataSet());
			double result = c.classifyInstance(instance);
			if(Double.compare(result, 1)==0){
				predictions[actionList.get(cnt)] = true;
			}
			else
				predictions[actionList.get(cnt)] = false;
			cnt++;
		}
		MultiLabelOutput mlo = new MultiLabelOutput(predictions);*/
		boolean []bipartition = new boolean[numLabels];
		double [] confidences = new double[numLabels];
		
		Instance tmpInstance = DataUtils.createInstance(instance, instance.weight(), instance.toDoubleArray());
		for(int i = 0; i<actionList.size(); i++){
			double distribution [];
			try {
                distribution = classifierChain.get(i).distributionForInstance(tmpInstance);
            } catch (Exception e) {
                System.out.println(e);
                return null;
            }
			int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;
			
			// Ensure correct predictions both for class values {0,1} and {1,0}
            Attribute classAttribute = classifierChain.get(i).getFilter().getOutputFormat().classAttribute();
            bipartition[actionList.get(i)] = (classAttribute.value(maxIndex).equals("1")) ? true : false;

            // The confidence of the label being equal to 1
            confidences[actionList.get(i)] = distribution[classAttribute.indexOfValue("1")];

            tmpInstance.setValue(labelIndices[actionList.get(i)], maxIndex);
		}
		MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
		//MultiLabelOutput mlo = ensemble.makePrediction(instance);
        return mlo;
	}

	@Override
	public TechnicalInformation getTechnicalInformation() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String globalInfo() {
		// TODO Auto-generated method stub
		return null;
	}

}
