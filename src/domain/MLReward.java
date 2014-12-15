package domain;

public class MLReward {

	public static double getHammingLoss(MLState s, double groundTruth[][]){
		double hammingLoss = 0;
		int symmetricDifference = 0;
		for(int i = 0; i< groundTruth.length; i++){
			for(int j = 0; j<groundTruth[0].length; j++){
				if(Double.compare(s.labelSpace[i][j], groundTruth[i][j])!=0)
					symmetricDifference++;
			}
			hammingLoss= hammingLoss+(double)symmetricDifference/groundTruth[0].length;
		}
		hammingLoss = (double) -hammingLoss/groundTruth.length;
		return hammingLoss;
	}
}
