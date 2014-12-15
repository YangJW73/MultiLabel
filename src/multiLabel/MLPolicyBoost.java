package multiLabel;
import java.util.ArrayList;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.TechnicalInformation;
import mulan.classifier.*;
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
	ArrayList<Classifier> classifierChain;
	ArrayList<Integer> actionList;
	int labelIndices[];
	MultiLabelInstances dataSet;
	
	public MLPolicyBoost(int iteration, int trialsPerIter, int maxStep, boolean isPara, double epsion){
		this.iteration = iteration;
		this.trialsPerIter = trialsPerIter;
		this.maxStep = maxStep;
		this.isPara = isPara;
		this.epsion = epsion;
	}

	@Override
	protected void buildInternal(MultiLabelInstances trainingSet)
			throws Exception {
		// TODO Auto-generated method stub
		Random random = new Random();
		numLabels = trainingSet.getNumLabels();
		dataSet = trainingSet;
		policy = new PolicyBoostDiscrete (new Random(random.nextInt()), Integer.MAX_VALUE);
		task = new MLTask(trainingSet, random);
		classifierChain = new ArrayList<Classifier>();
		actionList = new ArrayList<Integer>();
		labelIndices = trainingSet.getLabelIndices();
		MLState initialState = (MLState) task.getInitialState();
		
		MLExperiment mle = new MLExperiment();
		mle.trainPolicy(policy, task, iteration, trialsPerIter, initialState, maxStep, isPara, epsion, random);
		
		for(int i = 0; i<mle.bestTraj.getSamples().size(); i++){
			MLState tmpS = (MLState)mle.bestTraj.getSamples().get(i).sPrime;
			Action tmpA = mle.bestTraj.getSamples().get(i).action;
			classifierChain.add(tmpS.c);
			actionList.add(tmpA.a);
		}
	}

	@Override
	protected MultiLabelOutput makePredictionInternal(Instance instance)
			throws Exception, InvalidDataException {
		// TODO Auto-generated method stub
		boolean[] predictions = new boolean[numLabels];
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
		MultiLabelOutput mlo = new MultiLabelOutput(predictions);
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
