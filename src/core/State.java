/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package core;

/**
 * The abstract class for agents' state
 * @author qing da <daq@lamda.nju.edu.cn>
 */
public abstract class State {

    /**
     * The feature vector
     */
    protected double[] features;

    /**
     * The abstract method for feature extraction
     */
    protected abstract void extractFeature();

    /**
     * Get the feature vector
     *
     * @return The feature vector
     */
    public double[] getfeatures() {
        if (null == features) {
            extractFeature();
        }
        return features;
    }
}
