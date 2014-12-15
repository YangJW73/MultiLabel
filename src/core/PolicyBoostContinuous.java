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

import core.Action;
import core.State;
import core.Task;
import core.Trajectory;
import core.Tuple;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.trees.REPTree;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author qing da <daq@lamda.nju.edu.cn>
 */
public class PolicyBoostContinuous extends Policy {

    /**
     * The step size list
     */
    private List<Double>[] m_alphas;
    /**
     * The base learner list
     */
    private List<Classifier>[] m_potentialFunctions;
    /**
     * The base learner for regression
     */
    private Classifier m_base;
    /**
     * The step size
     */
    private double m_stepsize = 1;
    /**
     * The parameter of Gaussion policy
     */
    private double m_sigma = 0.05;
    /**
     * Data head of training instances
     */
    private Instances m_dataHead = null;
    /**
     * The pool of best trajectories
     */
    private Trajectory[] m_bestPool = null;
    /**
     * The pool of uniformly sampled trajectories
     */
    private Trajectory[] m_uniformPool = null;
    /**
     * The size of best pool
     */
    private int m_bestPoolSize = 10;
    /**
     * The size of uniform pool
     */
    private int m_uniformPoolSize = 10;
    /**
     * The current size of best pool
     */
    private int m_bestPoolCurSize = 0;
    /**
     * The current size of uniform pool
     */
    private int m_uniformPoolCurSize = 0;
    /**
     * The napping interval
     */
    private int m_nappingInterval = 0;
    /**
     * The number of threads for parallel training
     */
    private int m_threads;

    /**
     * The construction method
     *
     * @param rand The instance of the Random class
     * @param nappingInterval The napping interval
     */
    public PolicyBoostContinuous(Random rand, int nappingInterval) {
        m_numIteration = 0;

        m_random = rand;
        REPTree tree = new REPTree();
        tree.setMaxDepth(100);
        m_base = tree;

        m_bestPool = new Trajectory[m_bestPoolSize];
        m_uniformPool = new Trajectory[m_uniformPoolSize];

        this.m_nappingInterval = nappingInterval;
    }

    /**
     * set the current pool size of the best-so-far trajectories. Default is 10.
     *
     * @param size pool size in <code>int</code>
     */
    public void setPoolSizeOfBest(int size) {
        m_bestPoolSize = size;
    }

    /**
     * get the current pool size of the best-so-far trajectories
     *
     * @return pool size in <code>int</code>
     */
    public int getPoolSizeOfBest() {
        return m_bestPoolSize;
    }

    /**
     * set the current pool size of the uniformly sampled trajectories. Default
     * is 10.
     *
     * @param size pool size in <code>int</code>
     */
    public void setPoolSizeOfUniform(int size) {
        m_uniformPoolSize = size;
    }

    /**
     * get the current pool size of the uniformly sampled trajectories
     *
     * @return pool size in <code>int</code>
     */
    public int getPoolSizeOfUniform() {
        return m_uniformPoolSize;
    }

    /**
     * set the interval of iterations for applying the napping mechanism.
     * Default is 0 (no napping)
     *
     * @param i interval in <code>int</code>
     */
    public void setNappingInterval(int i) {
        m_nappingInterval = i;
    }

    /**
     * get the interval of iterations for applying the napping mechanism
     *
     * @return interval in <code>int</code>
     */
    public int getNappingInterval() {
        return m_nappingInterval;
    }

    /**
     * set the base learning algorithm. Default is the decision tree with 100
     * maximum depth.
     *
     * @param c an instance of weka.core.Classifier
     */
    public void setBaseLearner(Classifier c) {
        m_base = c;
    }

    /**
     * get the current base learning algorithm.
     *
     * @return an instance of weka.core.Classifier
     */
    public Classifier getBaseLearner() {
        Classifier c = null;
        try {
            c = Classifier.makeCopy(m_base);
        } catch (Exception ex) {
            Logger.getLogger(PolicyBoostDiscrete.class.getName()).log(Level.SEVERE, null, ex);
        }
        return c;
    }

    /**
     * set the parallelization of roll-outs. Default is 0 (no parallel).
     *
     * @param threads : number of threads to parallel. 0 means no parallel. -1
     * means use maximum cores available.
     */
    public void setParallelThreads(int threads) {
        m_threads = threads;
    }

    /**
     * get the parallelization threads of roll-outs.
     *
     * @return number of threads to parallel. 0 means no parallel. -1 means use
     * maximum cores available
     */
    public int getParallelThreads() {
        return m_threads;
    }

    /**
     * get the boosting step size
     *
     * @return step size in <code>double</code>
     */
    public double getStepsize() {
        return m_stepsize;
    }

    /**
     * set the boosting step size Default 1. Could be better being 0.1 and 0.01.
     *
     * @param stepsize
     */
    public void setStepsize(double stepsize) {
        this.m_stepsize = stepsize;
    }

    @Override
    public Action makeDecisionDeterministic(State s, Task t, Random outRand) {
        if (m_numIteration == 0) {
            return null;
        }

        double[] utilities = getUtility(s, t);
        if (utilities == null) {
            int ss = 1;
        }
        Action action = new Action(utilities);
        return new Action(action, 1);
    }

    @Override
    public Action makeDecisionStochastic(State s, Task t, Random outRand) {
        if (m_numIteration == 0) {
            return null;
        }

        Random thisRand = outRand == null ? m_random : outRand;

        double[] utilities = getUtility(s, t);
        double[] controls = sampleFromGaussian(utilities, thisRand);
        if (controls == null) {
            int ss = 1;
        }
        Action action = new Action(controls);
        return new Action(action, 1);
    }

    /**
     * Given the state and task, calcualte the utility value for each action
     *
     * @param s The state
     * @param t The task
     * @return The utility value for each action
     */
    public double[] getUtility(State s, Task t) {
        double[] utilities = new double[t.actionDim];
        for (int k = 0; k < utilities.length; k++) {
            double[] stateFeature = s.getfeatures();
            Instance ins = contructInstance(stateFeature, 0);
            if (null == m_dataHead) {
                m_dataHead = constructDataHead(stateFeature.length);
            }
            ins.setDataset(m_dataHead);
            utilities[k] = 0;
            if (m_potentialFunctions != null) {
                for (int j = 0; j < m_potentialFunctions[0].size(); j++) {
                    try {
                        utilities[k] += m_alphas[k].get(j) * m_potentialFunctions[k].get(j).classifyInstance(ins);
                    } catch (Exception ex) {
                        Logger.getLogger(PolicyBoostDiscrete.class.getName()).log(Level.SEVERE, null, ex);
                    }
                }
            }
        }

        return utilities;
    }

    /**
     * Construct instance from state feature vector and label
     *
     * @param stateFeature The feature vector
     * @param label The label
     * @return
     */
    private Instance contructInstance(double[] stateFeature, double label) {
        int D = stateFeature.length;
        double[] values = new double[D + 1];
        values[D] = label;
        System.arraycopy(stateFeature, 0, values, 0, D);
        Instance ins = new Instance(1.0, values);
        return ins;
    }

    /**
     * Construct data head of instances
     *
     * @param dim The dimension of feature
     * @return Instances
     */
    public Instances constructDataHead(int dim) {
        FastVector attInfo_x = new FastVector();
        for (int i = 0; i < dim; i++) {
            attInfo_x.addElement(new Attribute("att_" + i, i));
        }

        attInfo_x.addElement(new Attribute("class", dim));
        Instances data = new Instances("dataHead", attInfo_x, 0);
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    @Override
    public void update(List<Trajectory> trajectories) {
        // gather data to learn
        List<Trajectory> trainTrajectories = new ArrayList<Trajectory>();
        // current data
        for (int i = 0; i < trajectories.size(); i++) {
            trainTrajectories.add(trajectories.get(i));
        }
        // best data
        for (int i = 0; i < m_bestPoolCurSize; i++) {
            trainTrajectories.add(m_bestPool[i]);
        }
        // uniform data
        for (int i = 0; i < m_uniformPoolCurSize; i++) {
            trainTrajectories.add(m_uniformPool[i]);
        }

        int numZ = trainTrajectories.size();

        double[] P_z = new double[numZ];
        compuate_P_z(trainTrajectories, P_z);

        double sumR = 0;
        double sumP = 0;
        for (int i = 0; i < numZ; i++) {
            Trajectory trajectory_i = trainTrajectories.get(i);
            sumR += trajectory_i.getRewards();
            sumP += P_z[i];
        }

        double max_abs_label = -1;
        for (int i = 0; i < trainTrajectories.size(); i++) {
            Trajectory trajectory = trainTrajectories.get(i);
            Task task = trajectory.getTask();
            List<Tuple> samples = trajectory.getSamples();

            List<double[]> features = new ArrayList<double[]>();
            List<Double>[] multiModelLabels = new List[task.actionDim];
            for (int k = 0; k < task.actionDim; k++) {
                multiModelLabels[k] = new ArrayList<Double>();
            }

            double R_z = trajectory.getRewards();

            for (int step = 0; step < samples.size(); step++) {
                Tuple sample = samples.get(step);

                features.add(sample.s.getfeatures());
                double prab = ((Action) sample.action).probability;

                for (int k = 0; k < task.actionDim; k++) {
                    double[] rhos = getUtility(sample.s, task);
                    double label = ((numZ * R_z - sumR) / prab + (numZ * P_z[i] - sumP) * sample.reward) * prab * 2 * (sample.action.controls[k] - rhos[k]) / (m_sigma * m_sigma);
                    multiModelLabels[k].add(label);

                    if (Math.abs(label) > max_abs_label) {
                        max_abs_label = Math.abs(label);
                    }
                }
            }

            trajectory.setFeatures(features);
            trajectory.setMultiModelLabels(multiModelLabels);
        }

        if (null == m_alphas) {
            int actionDim = trainTrajectories.get(0).getTask().actionDim;
            m_alphas = new List[actionDim];
            m_potentialFunctions = new List[actionDim];
            for (int k = 0; k < actionDim; k++) {
                m_alphas[k] = new ArrayList<Double>();
                m_potentialFunctions[k] = new ArrayList<Classifier>();
            }
        }

        ExecutorService exec = Executors.newFixedThreadPool(
                Runtime.getRuntime().availableProcessors() - 1);
        List<Instances> dataList = new ArrayList<Instances>();
        for (int k = 0; k < trajectories.get(0).getTask().actionDim; k++) {
            Instances data = new Instances(m_dataHead);
            for (Trajectory trajectory : trainTrajectories) {
                List<double[]> features = trajectory.getFeatures();
                List<Double>[] multiModelLabels = trajectory.getMultiModelLabels();
                for (int j = 0; j < features.size(); j++) {
                    Instance ins = contructInstance(features.get(j), multiModelLabels[k].get(j) / max_abs_label);
                    data.add(ins);
                }
            }
            dataList.add(data);

            Classifier c = getBaseLearner();
            try {
                ParallelTrain run = new ParallelTrain(c, data);
                exec.execute(run);
            } catch (Exception ex) {
                Logger.getLogger(PolicyBoostDiscrete.class.getName()).log(Level.SEVERE, null, ex);
            }

            m_alphas[k].add(m_stepsize);

            m_potentialFunctions[k].add(c);
        }
        exec.shutdown();
        try {
            while (!exec.awaitTermination(10, TimeUnit.SECONDS)) {
            }
        } catch (InterruptedException ex) {
            Logger.getLogger(PolicyBoostDiscrete.class.getName()).log(Level.SEVERE, null, ex);
        }

        m_numIteration++;

        if (m_nappingInterval > 0 && m_numIteration > 0 && (m_numIteration + 1) % m_nappingInterval == 0) {
            System.out.println("napping...(model=" + m_potentialFunctions[0].size() + ")");
            napping(dataList);
            System.out.println("napped...(model=" + m_potentialFunctions[0].size() + ")");
        }
    }

    /**
     * The napping operation
     *
     * @param dataList The list of probing instances
     */
    private void napping(List<Instances> dataList) {
        for (int k = 0; k < m_potentialFunctions.length; k++) {
            Instances data = dataList.get(k);

            for (int i = 0; i < data.numInstances(); i++) {
                Instance ins = data.instance(i);
                double val = 0;
                for (int j = 0; j < m_alphas[k].size(); j++) {
                    try {
                        val += m_alphas[k].get(j) * m_potentialFunctions[k].get(j).classifyInstance(ins);
                    } catch (Exception ex) {
                        Logger.getLogger(PolicyBoostDiscrete.class.getName()).log(Level.SEVERE, null, ex);
                    }
                }
                ins.setClassValue(val);
            }

            REPTree reptree = new REPTree();
            reptree.setMaxDepth(100);

            Classifier c = reptree;
            try {
                c.buildClassifier(data);
            } catch (Exception ex) {
                Logger.getLogger(PolicyBoostDiscrete.class.getName()).log(Level.SEVERE, null, ex);
            }
            m_alphas[k].clear();
            m_potentialFunctions[k].clear();

            m_alphas[k].add(1.0);
            m_potentialFunctions[k].add(c);
        }
    }

    /**
     * Calculate the log probability of trajectories
     *
     * @param trainTrajectories The list of trajectories
     * @param log_P_z The log probability to be returned
     */
    private void compuate_P_z(List<Trajectory> trainTrajectories, double[] log_P_z) {
        int numTraj = trainTrajectories.size();
        double min = Double.POSITIVE_INFINITY;
        for (int i = 0; i < numTraj; i++) {
            Trajectory trajectory = trainTrajectories.get(i);

            log_P_z[i] = 0;
            int T = trajectory.getSamples().size();
            for (int t = 0; t < T; t++) {
                Tuple tuple = trajectory.getSamples().get(t);
                double probability = getProbability(tuple.s, tuple.action, trajectory.getTask());
                log_P_z[i] += Math.log(probability);
            }

            if (log_P_z[i] < min) {
                min = log_P_z[i];
            }
        }

        for (int i = 0; i < numTraj; i++) {
            log_P_z[i] = log_P_z[i] / 1000;
        }
    }

    /**
     * Calculate the probabilities of action
     *
     * @param s The state
     * @param action The action
     * @param task The task
     * @return The probabilities of each action
     */
    private double getProbability(State s, Action action, Task task) {
        double[] utilities = getUtility(s, task);
        double probability = 1, prob = 1 / (Math.sqrt(Math.PI * 2) * m_sigma);
        for (int i = 0; i < utilities.length; i++) {
            prob = prob * Math.exp(-(action.controls[i] - utilities[i]) * (action.controls[i] - utilities[i]) / (m_sigma * m_sigma));
        }

        // The experiments show that directly returning 1 as probability always results in better performance
        return probability;
    }

    /**
     * Sample continuous actions from the utility value
     *
     * @param utilities The utility value
     * @param thisRand An instance of Random class
     * @return The sampled actions
     */
    private double[] sampleFromGaussian(double[] utilities, Random thisRand) {
        double[] controls = new double[utilities.length];

        for (int i = 0; i < controls.length; i++) {
            controls[i] = utilities[i] + thisRand.nextGaussian() * m_sigma;
        }

        return controls;
    }
}
