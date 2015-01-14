package domain;
import weka.classifiers.*;
import weka.core.Instances;
import core.State;

public class MLState extends State{

	public double labelSpace[][];
	public double labelDis[];
	public Classifier c;
	
	public MLState(double newLabelSpace[][], int labelNum, Classifier c){
		this.c = c;
		labelSpace = newLabelSpace;
		labelDis = new double [labelNum];
		for(int i = 0; i<newLabelSpace.length; i++){
			for(int j = 0; j<newLabelSpace[0].length; j++){
				labelDis[j] += labelSpace[i][j];
			}
		}
	}
	
	public MLState(int instNum, int labelNum){
		labelSpace = new double[instNum][labelNum];
		labelDis = new double[labelNum];
		for(int j = 0; j<labelSpace[0].length; j++){
			labelDis[j] = 0;
			for(int i = 0; i<labelSpace.length; i++){
				labelSpace[i][j] = 0;
			}
		}
		
	}

	@Override
	protected void extractFeature() {
		// TODO Auto-generated method stub
		features = new double[labelDis.length];
		for(int j = 0; j<labelDis.length; j++){
			features[j] += labelDis[j];
		}
	}
}
