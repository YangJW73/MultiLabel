package domain;

public class MLReward {

	public static double getHammingLoss(MLState s, double groundTruth[][]){
		double hammingLoss = 0;
		
		double h1 = 0;
		double h2 = 0;
		for(int i = 0; i<groundTruth[0].length; i++){
			int symmetricDifference = 0;
			for(int j = 0; j<groundTruth.length; j++){
				if(s.isPredicted[j]){
					if(Double.compare(s.labelSpace[j][i], groundTruth[j][i])!=0)
						symmetricDifference++;
				}
				else
					symmetricDifference++;
			}
			h1= h1+(double)symmetricDifference/groundTruth.length;
		}
		
		h1 = (double) -h1/groundTruth[0].length;
		
		for(int i = 0; i<groundTruth[0].length; i++){
			int symmetricDifference = 0;
			for(int j = 0; j<groundTruth.length; j++){
				if(s.isPredicted[j]){
					if(Double.compare(s.labelSpace[j][i], groundTruth[j][i])!=0)
						symmetricDifference++;
				}
				else
					symmetricDifference++;
			}
			h2= h2+(double)symmetricDifference/groundTruth.length;
		}
		h2 = (double) -h2/groundTruth[0].length;
		hammingLoss = h1 - h2;
		return hammingLoss;
	}
	
	public static double getSubsetAccuracy(MLState s, double groundTruth[][]){
		double subacc = 0;
		for(int i = 0; i<groundTruth.length; i++){
			boolean isSame = true;
			for(int j = 0; j<groundTruth[0].length; j++){
				if(Double.compare(s.labelSpace[i][j], groundTruth[i][j]) != 0)
					isSame = false;
			}
			if(isSame)
				subacc++;
		}
		subacc = subacc/groundTruth[0].length;
		return subacc;
	}
	
	public static double getAccuracy(MLState s, double groundTruth[][]){
		double accuracy = 0;
		
		for(int i = 0; i<groundTruth[0].length; i++){
			double intersection = 0;
			double union = 0;
			for(int j = 0; j<groundTruth.length; j++){
				if(s.isPredicted[j]){
					if((Double.compare(s.labelSpace[j][i], 1) == 0)&&(Double.compare(groundTruth[j][i], 1) == 0))
						intersection++;
				}
				if((Double.compare(s.labelSpace[j][i], 1) == 0)||(Double.compare(groundTruth[j][i], 1) == 0))
					union++;
				
			}
			accuracy += intersection/union;
		}
		accuracy = accuracy/groundTruth[0].length;
		return accuracy;
	}
}
