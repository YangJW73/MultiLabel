/* This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 *
 * Copyright (C) 2014 Nanjing University, Nanjing, China
 */
package core;

import core.Task;
import java.util.List;

/**
 * The class which stores all information about one trajectory
 *
 * @author qing da <daq@lamda.nju.edu.cn>
 */
public class Trajectory implements Comparable<Trajectory> {

    private Task task;
    private List<Tuple> samples;
    private double rewards;
    private int maxStep;
    private boolean isSuccess;
    List<double[]> features;
    List<Double> labels;
    List<Double>[] multiModelLabels;
    private int producedIteration;

    /**
     * The construction method
     *
     * @param task The task
     * @param samples The data
     * @param maxStep Maximal step of once execution
     * @param isSuccess Whether the trajectory is successful
     */
    public Trajectory(Task task, List<Tuple> samples, int maxStep, boolean isSuccess) {
        this.task = task;
        this.samples = samples;
        this.maxStep = maxStep;
        this.isSuccess = isSuccess;
    }

    /**
     * Return the task
     *
     * @return The task
     */
    public Task getTask() {
        return task;
    }

    /**
     * Get samples
     *
     * @return
     */
    public List<Tuple> getSamples() {
        return samples;
    }

    /**
     * Get the rewards of this trajectory
     *
     * @return The rewards of this trajectory
     */
    public double getRewards() {
        return rewards;
    }

    /**
     * Get the rewards of fixed step
     *
     * @return The rewards of fixed step
     */
    public double getRZ() {
        double RZ = 0;
        for (int i = 0; i < samples.size(); i++) {
            RZ += (i + 1) * samples.get(i).reward;
        }
        if (isSuccess) {
            RZ += samples.get(samples.size() - 1).reward * (maxStep - (samples.size() * (samples.size() - 1)) / 2);
        } else {
            RZ += rewards * (samples.size() - 1) / 2;
        }
        return RZ;
    }

    /**
     * Set the rewards
     *
     * @param rewards The rewards
     */
    public void setRewards(double rewards) {
        this.rewards = rewards;
    }

    /**
     * Return whether the trajectory is successful
     *
     * @return Whether the trajectory is successful
     */
    public boolean isIsSuccess() {
        return isSuccess;
    }

    /**
     * Set whether the trajectory is successful
     *
     * @param isSuccess Whether the trajectory is successful
     */
    public void setIsSuccess(boolean isSuccess) {
        this.isSuccess = isSuccess;
    }

    /**
     * Set the feature vectors of this trajectory
     *
     * @param features The feature vectors of this trajectory
     */
    public void setFeatures(List<double[]> features) {
        this.features = features;
    }

    /**
     * Set the labels of correspingding feature vectors
     *
     * @param labels
     */
    public void setLabels(List<Double> labels) {
        this.labels = labels;
    }

    /**
     * Get the labels of correspingding feature vectors
     *
     * @return
     */
    public List<Double> getLabels() {
        return labels;
    }

    /**
     * Set the feature vectors
     *
     * @return
     */
    public List<double[]> getFeatures() {
        return features;
    }

    /**
     * Get the iteration at which the trajectory is produced
     *
     * @return The iteration at which the trajectory is produced
     */
    public int getProducedIteration() {
        return producedIteration;
    }

    /**
     * Set the iteration at which the trajectory is produced
     *
     * @param producedIteration The iteration at which the trajectory is
     * produced
     */
    public void setProducedIteration(int producedIteration) {
        this.producedIteration = producedIteration;
    }

    /**
     * Get labels of multi-model data (for continuous domain with multiple
     * control dimensions)
     *
     * @return
     */
    public List<Double>[] getMultiModelLabels() {
        return multiModelLabels;
    }

    /**
     * Set labels of multi-model data
     *
     * @param multiModelLabels
     */
    public void setMultiModelLabels(List<Double>[] multiModelLabels) {
        this.multiModelLabels = multiModelLabels;
    }

    @Override
    public int compareTo(Trajectory o) {
        return new Double(o.getRewards()).compareTo(this.getRewards());
    }
}
