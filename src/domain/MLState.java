package domain;
import java.util.ArrayList;

import weka.classifiers.*;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import core.State;

public class MLState extends State{

	public double labelSpace[][];
	public double preLabelSpace[][];
	public double labelDis[];
	public boolean isPredicted[];
	public boolean isPredicted_Pre[];
	public FilteredClassifier c;
	
	public MLState(double newLabelSpace[][], boolean chain[], int labelNum, FilteredClassifier c){
		this.c = c;
		preLabelSpace = labelSpace;
		labelSpace = newLabelSpace;
		isPredicted_Pre = isPredicted;
		labelDis = new double [labelNum];
		isPredicted = chain;
		/*for(int i = 0; i<newLabelSpace.length; i++){
			for(int j = 0; j<newLabelSpace[0].length; j++){
				labelDis[j] += labelSpace[i][j];
			}
		}*/
		for(int j = 0; j<newLabelSpace.length; j++){
			for(int i = 0; i<newLabelSpace[0].length; i++){
				labelDis[j] += labelSpace[j][i];
			}
		}
	}
	
	public MLState(int instNum, int labelNum){
		/*labelSpace = new double[instNum][labelNum];
		labelDis = new double[labelNum];
		for(int j = 0; j<labelSpace[0].length; j++){
			labelDis[j] = 0;
			for(int i = 0; i<labelSpace.length; i++){
				labelSpace[i][j] = 0;
			}
		}*/
		labelSpace = new double[labelNum][instNum];
		preLabelSpace = new double[labelNum][instNum];
		labelDis = new double[labelNum];
		isPredicted = new boolean [labelNum];
		isPredicted_Pre = new boolean [labelNum];
		for(int j = 0; j<labelSpace.length; j++){
			labelDis[j] = 0;
			isPredicted[j] = false;
			isPredicted_Pre[j] = false;
			for(int i = 0; i<labelSpace[0].length; i++){
				labelSpace[j][i] = 0;
				preLabelSpace[j][i] = 0;
			}
		}
	}
	
	double binaryToDec(double v[]){
		double result = 0 ;
		for(int i = 0; i<v.length; i++){
			result += Math.pow(2, i)*v[i];
		}
		return result;
	}
	
	double computeCosine(double a[], double b[]){
		double result = 0;
		double tmp1 = 0;
		double tmp2 = 0;
		double tmp3 = 0;
		for(int i = 0; i<a.length; i++){
			tmp1 += a[i]*b[i];
			tmp2 += a[i]*a[i];
			tmp3 += b[i]*b[i];
		}
		result = tmp1/(Math.sqrt(tmp2)*Math.sqrt(tmp3));
		return result;
	}
	
	double computePearson(double a[], double b[]){
		double result = 0;
		double tmp1 = 0;
		double tmp2 = 0;
		double tmp3 = 0;
		double tmp4 = 0;
		double tmp5 = 0;
		for(int i = 0; i<a.length; i++){
			tmp1 = tmp1 + a[i]*b[i];
			tmp2 = tmp2 + a[i];
			tmp3 = tmp3 + b[i];
			tmp4 = tmp4 + a[i]*a[i];
			tmp5 = tmp5 + b[i]*b[i];
		}
		
		result = (tmp1*a.length - tmp2*tmp3)/(Math.sqrt(a.length*tmp4-tmp2*tmp2)*Math.sqrt(a.length*tmp5-tmp3*tmp3));
		if(Double.isNaN(result))
			result = 0;
		return result;
	}

	@Override
	protected void extractFeature() {
		// TODO Auto-generated method stub
		/*
		 * only label histogram
		 */
		/*features = new double[labelDis.length];
		for(int j = 0; j<labelDis.length; j++){
			features[j] += labelDis[j];
		}*/
		/*
		 * label histogram & position
		 */
		ArrayList<Double> featVector = new ArrayList<Double>();
		
		for(int j = 0; j<labelDis.length; j++){
			featVector.add(labelDis[j]/this.labelSpace[0].length);
		}
		//double total = (double)(Math.pow(2, labelDis.length)-1)*(labelSpace.length-1)*labelSpace.length/2;
		
		/*double tmp = 0;
		for(int i = 0; i<labelSpace.length; i++){
			tmp += binaryToDec(labelSpace[i])*i;
		}
		features[labelDis.length] += tmp/total;*/
		
		for(int i = 0; i<this.labelSpace.length; i++){
			for(int j = i+1; j<this.labelSpace.length; j++){
				featVector.add(computePearson(labelSpace[i], labelSpace[j]));
			}
		}
		features = new double[featVector.size()];
		for(int i = 0; i<featVector.size(); i++){
			features[i] = featVector.get(i);
		}
		//System.out.println("test");
	}
}
