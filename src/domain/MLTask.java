package domain;

import java.util.*;

import mulan.data.MultiLabelInstances;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.*;
import core.Action;
import core.State;
import core.Task;

public class MLTask extends Task{
	int instNum;
	int labelNum;
	Random random;
	double groundTruth[][];
	int labelIndices[];
	Instances trainingData;
	
	
	public MLTask(MultiLabelInstances dataSet, Random random){
		this.instNum = dataSet.getNumInstances();
		this.labelNum = dataSet.getNumLabels();
		this.random = random;
		this.actions = new Action[labelNum];
		this.groundTruth = new double[instNum][labelNum];
		Instances data = dataSet.getDataSet();
		this.trainingData = new Instances(dataSet.getDataSet());
		this.labelIndices = dataSet.getLabelIndices();
		for(int i = 0; i<instNum; i++){
			for(int j = 0; j<labelNum; j++){
				this.groundTruth[i][j] = data.instance(i).value(labelIndices[j]);
			}
		}
		for(int a = 0; a<labelNum; a++)
			actions[a] = new Action(a);
	}

	@Override
	public State getInitialState() {
		// TODO Auto-generated method stub
		return new MLState(instNum, labelNum);
	}

	double [][] setNewLabel(MLState s, Classifier c, int labelIndex){
		double newLabelSpace[][] = new double[this.instNum][this.labelNum];
		for(int i = 0; i<trainingData.numInstances(); i++){
			for(int j = 0; j<this.labelNum; j++){
				if(this.labelIndices[j] == labelIndex){
					try {
						c.classifyInstance(trainingData.instance(i));
						newLabelSpace[i][j] = c.classifyInstance(trainingData.instance(i));
					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
				else{
					newLabelSpace[i][j] = s.labelSpace[i][j];
				}
			}	
		}
		return newLabelSpace;
	}
	@Override
	public State transition(State s, Action a, Random outRand) {
		// TODO Auto-generated method stub
		Random thisRand = outRand == null ? random : outRand;
		MLState mls = (MLState) s;
		int move = a.a;
		if(move<labelNum){
			trainingData.setClassIndex(labelIndices[move]);
			J48 dt = new J48();
			try {
				dt.buildClassifier(trainingData);
				//baseClassifiers.add(dt);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			return new MLState(setNewLabel(mls,dt,labelIndices[move]),this.labelNum, dt);
		}
		else if(move == labelNum){
			
		}
		return null;
	}

	@Override
	public double immediateReward(State s) {
		// TODO Auto-generated method stub
		MLState mls = (MLState) s;
		return MLReward.getHammingLoss(mls, groundTruth);
	}

	@Override
	public boolean isComplete(State s) {
		// when the predict set is exactly same with ground truth
		boolean isSame = true;
		MLState mls = (MLState) s;
		for(int i = 0; i<instNum; i++){
			for(int j = 0; j<labelNum; j++){
				if(Double.compare(mls.labelSpace[i][j], groundTruth[i][j]) != 0)
					isSame = false;
			}
		}
		if(isSame)
			return true;
		else	
			return false;
	}

}
