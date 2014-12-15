package domain;
import weka.classifiers.*;
import weka.core.Instances;
import core.State;

public class MLState extends State{

	public double labelSpace[][];
	public Classifier c;
	
	public MLState(double newLabelSpace[][], int labelNum, Classifier c){
		this.c = c;
		labelSpace = new double[newLabelSpace.length][labelNum];
		for(int i = 0; i<labelSpace.length; i++){
			for(int j = 0; j<labelSpace[0].length; j++){
				labelSpace[i][j] = newLabelSpace[i][j];
			}
		}
	}
	
	public MLState(int instNum, int labelNum){
		labelSpace = new double[instNum][labelNum];
		for(int i = 0; i<labelSpace.length; i++){
			for(int j = 0; j<labelSpace[0].length; j++){
				labelSpace[i][j] = 0.5;
			}
		}
	}

	@Override
	protected void extractFeature() {
		// TODO Auto-generated method stub
		features = new double[labelSpace.length*labelSpace[0].length];
		int cnt = 0;
		for(int i = 0; i<labelSpace.length; i++){
			for(int j = 0; j<labelSpace[0].length; j++){
				features[cnt] = labelSpace[i][j];
				cnt++;
			}
		}
	}
}
