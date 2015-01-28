package core;

import core.Action;
import core.State;
import core.Task;
import core.Trajectory;
import core.Tuple;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 * The PolicyBoost policy for discrete domain
 *
 * @author qing da <daq@lamda.nju.edu.cn>
 * revised May 23, 2014, by
 * @author Yang Yu <eyounx@gmail.com>
 */
public class PolicyBoostDiscrete extends GibbsPolicy {

    /**
     * The flag list indicating whether a base learner is a regressor or
     * classifier
     */
    private List<Boolean> m_types;
    /**
     * The step size list
     */
    private List<Double> m_alphas;
    /**
     * The base learner list
     */
    private List<Classifier> m_potentialFunctions;
    /**
     * The base learner for regression and classification
     */
    private Classifier m_base, m_napBase;
    /**
     * The step size
     */
    private double m_stepsize = 1;
    /**
     * Data head of training instances
     */
    private Instances m_dataHead = null, m_dataHead2 = null, m_nappingPool = null;
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
     * The counter for reservoir sampling
     */
    private int m_uniformPoolCount = 0;
    /**
     * The number of threads for parallel training
     */
    private int m_threads = 0;
    /**
     * The napping interval
     */
    private int m_nappingInterval = 0;
    /**
     * The buffer size of reservoir sampling
     */
    private int m_nappingSize = 5000;
    /**
     * The counter for napping
     */
    private int m_nappingCount = 0;

    /**
     * The construction method
     *
     * @param rand The instance of the Random class
     * @param nappingInterval The napping interval
     */
    public PolicyBoostDiscrete(Random rand, int nappingInterval) {
        m_numIteration = 0;
        m_types = new ArrayList<Boolean>();
        m_alphas = new ArrayList<Double>();
        m_potentialFunctions = new ArrayList<Classifier>();
        m_random = rand;

        REPTree tree = new REPTree();
        tree.setMaxDepth(100);
        m_base = tree;

        RandomForest rf = new RandomForest();
        rf.setMaxDepth(100);
        rf.setSeed(rand.nextInt());
        m_napBase = rf;

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
            c = AbstractClassifier.makeCopy(m_base);
        } catch (Exception ex) {
            Logger.getLogger(PolicyBoostDiscrete.class.getName()).log(Level.SEVERE, null, ex);
        }

        return c;
    }

    /**
     * get the current base learning algorithm for napping.
     *
     * @return an instance of weka.core.Classifier
     */
    public Classifier getNappingBaseLearner() {
        Classifier c = null;
        try {
            c = AbstractClassifier.makeCopy(m_napBase);
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
     * Step size of boosting. 1 by default. Could be better to set to 0.1 and
     * 0.01.
     *
     * @param stepsize
     */
    public void setStepsize(double stepsize) {
        this.m_stepsize = stepsize;
    }

    @Override
    public void setNumIteration(int numIteration) {
        this.m_numIteration = Math.min(m_potentialFunctions.size(), numIteration);
    }

    @Override
    public Action makeDecisionDeterministic(State s, Task t, Random outRand) {
        if (m_numIteration == 0) {
            return null;
        }

        Random thisRand = outRand == null ? m_random : outRand;
        int K = t.actions.length;

        // return the action with the hightest probability
        double[] probabilities = getProbability(s, t);
        int bestAction = 0, m = 2;
        for (int k = 1; k < K; k++) {
            double diff = probabilities[k] - probabilities[bestAction];
            if (diff > Double.MIN_VALUE) {
                bestAction = k;
                m = 2;
            } else if (diff <= Double.MIN_VALUE && diff >= Double.MIN_VALUE) {
                // ties are broken randomly
                if (thisRand.nextDouble() < 1.0 / m) {
                    bestAction = k;
                }
                m++;
            }
        }

        return new Action(bestAction);
    }

    @Override
    public Action makeDecisionStochastic(State s, Task t, Random outRand) {
        if (m_numIteration == 0) {
            return null;
        }

        Random thisRand = outRand == null ? m_random : outRand;

        double[] probabilities = getProbability(s, t);
        return makeDecisionStochastic(s, t, probabilities, thisRand);
    }

    @Override
    public Action makeDecisionStochastic(State s, Task t, double[] probabilities, Random outRand) {
        if (m_numIteration == 0 || probabilities == null) {
            return null;
        }

        Random thisRand = outRand == null ? m_random : outRand;
        int K = t.actions.length;

        int bestAction = 0, m = 2;
        for (int k = 1; k < K; k++) {
            if (probabilities[k] > probabilities[bestAction] + Double.MIN_VALUE) {
                bestAction = k;
                m = 2;
            } else if (Math.abs(probabilities[k] - probabilities[bestAction]) <= Double.MIN_VALUE) {
                if (thisRand.nextDouble() < 1.0 / m) {
                    bestAction = k;
                }
                m++;
            }
        }

        return new Action(bestAction, probabilities[bestAction]);
    }
    
    @Override
    public Action makeDecisionStochastic(State s, Task t, double[] probabilities, ArrayList<Integer> usedAction, Random outRand){
    	if (m_numIteration == 0 || probabilities == null) {
            return null;
        }

        Random thisRand = outRand == null ? m_random : outRand;
        int K = t.actions.length;

        ArrayList<Integer> availabelAction = new ArrayList<Integer>();
        for(int k = 0; k < K; k++){
        	if(!usedAction.contains(k))
        		availabelAction.add(k);
        }
        int bestAction = availabelAction.get(0), m = 2;
        for (int k = 0; k < availabelAction.size(); k++) {
            if (probabilities[availabelAction.get(k)] > probabilities[bestAction] + Double.MIN_VALUE) {
                bestAction = availabelAction.get(k);
                m = 2;
            } else if (Math.abs(probabilities[availabelAction.get(k)] - probabilities[bestAction]) <= Double.MIN_VALUE) {
                if (thisRand.nextDouble() < 1.0 / m) {
                    bestAction = availabelAction.get(k);
                }
                m++;
            }
        }

        return new Action(bestAction, probabilities[bestAction]);
    }

    @Override
    public double[] getUtility(State s, Task t) {
        int A = t.actions.length;
        double[] utilities = new double[A];
        double maxUtility = Double.NEGATIVE_INFINITY;
        double[] stateActionFeature = t.getSAFeature(s, new Action(0));
        Instance ins = contructInstance(stateActionFeature, 0);
        calculatePV(ins, A, stateActionFeature.length, utilities);

        for (int a = 0; a < A; a++) {
            if (utilities[a] > maxUtility) {
                maxUtility = utilities[a];
            }
        }

        double norm = 0;
        for (int a = 0; a < A; a++) {
            utilities[a] = Math.exp(utilities[a]);
            norm += utilities[a];
        }

        for (int a = 0; a < A; a++) {
            utilities[a] /= norm;
        }

        return utilities;
    }

    /**
     * Construct instance from state and action feature vector and label
     *
     * @param stateActionTaskFeature The feature vector
     * @param label The label
     * @return
     */
    private Instance contructInstance(double[] stateActionTaskFeature, double label) {
        int D = stateActionTaskFeature.length;
        double[] values = new double[D + 1];
        values[D] = label;
        System.arraycopy(stateActionTaskFeature, 0, values, 0, D);
        DenseInstance ins = new DenseInstance(1.0, values);
        return ins;
    }

    /**
     * Construct data head of instances
     *
     * @param D The dimension of feature
     * @param na The number of action
     * @return Instances
     */
    public Instances constructDataHead(int D, int na) {
    	ArrayList<Attribute> attInfo_x = new ArrayList<Attribute>();
        for (int i = 0; i < D - 1; i++) {
            attInfo_x.add(new Attribute("att_" + i, i));
        }

        ArrayList<String> att = new ArrayList<String>(na);
        for (int i = 0; i < na; i++) {
            att.add("" + i);
        }
        attInfo_x.add(new Attribute("action", att, D - 1));

        attInfo_x.add(new Attribute("class", D));
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
        for (int i = 0; i < numZ; i++) {
            List<double[]> features = new ArrayList<double[]>();
            List<Double> labels = new ArrayList<Double>();

            Trajectory trajectory = trainTrajectories.get(i);
            Task task = trajectory.getTask();
            List<Tuple> samples = trajectory.getSamples();

            double R_z = trajectory.getRewards();

            for (int step = 0; step < samples.size(); step++) {
                Tuple sample = samples.get(step);

                features.add(task.getSAFeature(sample.s, sample.action));
                double prab = ((Action) sample.action).probability;

                double label = ((numZ * R_z - sumR) + (numZ * P_z[i] - sumP) * sample.reward * prab) * (1 - prab);
                if (Double.isNaN(label)) {
                    label = 1;
                }
                labels.add(label);

                if (Math.abs(label) > max_abs_label) {
                    max_abs_label = Math.abs(label);
                }
            }

            trajectory.setFeatures(features);
            trajectory.setLabels(labels);
        }

        Instances data = new Instances(m_dataHead);
        for (Trajectory trajectory : trainTrajectories) {
            List<double[]> features = trajectory.getFeatures();
            List<Double> labels = trajectory.getLabels();
            for (int j = 0; j < features.size(); j++) {
                Instance ins = contructInstance(features.get(j), max_abs_label == 0 ? labels.get(j) : labels.get(j) / max_abs_label);
                data.add(ins);

                if (m_nappingPool.numInstances() < m_nappingSize) {
                    m_nappingPool.add(ins);
                } else if (m_random.nextDouble() < m_nappingSize * 1.0 / m_nappingCount) {
                    m_nappingPool.delete(m_random.nextInt(m_nappingPool.numInstances()));
                    m_nappingPool.add(ins);
                }
                m_nappingCount++;
            }
        }

        Classifier c = getBaseLearner();
        try {
            c.buildClassifier(data);
        } catch (Exception ex) {
            Logger.getLogger(PolicyBoostDiscrete.class.getName()).log(Level.SEVERE, null, ex);
        }

        // sample for best and uniform
        {
            // best
            Trajectory[] allTrajectory = new Trajectory[trajectories.size() + m_bestPoolCurSize];
            trajectories.toArray(allTrajectory);
            System.arraycopy(m_bestPool, 0, allTrajectory, trajectories.size(), m_bestPoolCurSize);
            Arrays.sort(allTrajectory);
            m_bestPoolCurSize = Math.min(m_bestPoolSize, allTrajectory.length);
            System.arraycopy(allTrajectory, 0, m_bestPool, 0, m_bestPoolCurSize);

            for (Trajectory trajectory : trajectories) {
                // uniform
                if (m_uniformPoolCurSize < m_uniformPoolSize) {
                    m_uniformPool[m_uniformPoolCurSize] = trajectory;
                    m_uniformPoolCurSize++;
                } else {
                    int repInd = m_random.nextInt(m_uniformPoolCount);
                    if (repInd < m_uniformPoolSize) {
                        m_uniformPool[repInd] = trajectory;
                    }
                }
                m_uniformPoolCount++;
            }
        }

        m_types.add(Boolean.TRUE);
        m_potentialFunctions.add(c);
        m_alphas.add(1.0);

        if (m_nappingInterval > 0 && m_numIteration > 0 && (m_numIteration) % m_nappingInterval == 0) {
            System.out.println("napping...(model=" + m_potentialFunctions.size() + ")");
            napping(m_nappingPool, trainTrajectories.get(0).getTask().actions.length);
            System.out.println("napped...(model=" + m_potentialFunctions.size() + ")");
        }

        m_numIteration++;
    }

    /**
     * The napping operation
     *
     * @param data The probing instances
     * @param actionDim The number of actions
     */
    private void napping(Instances data, int actionDim) {
        int NS = data.numInstances();
        Instances sampledData = new Instances(m_dataHead2, NS);
        for (int i = 0; i < NS; i++) {
            Instance ins = data.instance(i);
            double maxUtility = Double.NEGATIVE_INFINITY;
            int maxInd = -1;
            double[] utilities = new double[actionDim];
            calculatePV(ins, actionDim, data.numAttributes() - 1, utilities);

            for (int a = 0; a < actionDim; a++) {
                if (utilities[a] > maxUtility) {
                    maxUtility = utilities[a];
                    maxInd = a;
                }
            }

            double[] fea = new double[data.numAttributes() - 1];
            for (int k = 0; k < fea.length - 1; k++) {
                fea[k] = ins.value(k);
            }
            fea[fea.length - 1] = maxInd;
            DenseInstance ins2 = new DenseInstance(1, fea);
            sampledData.add(ins2);
        }

        Classifier c = getNappingBaseLearner();
        try {
            c.buildClassifier(sampledData);
        } catch (Exception ex) {
            Logger.getLogger(PolicyBoostDiscrete.class.getName()).log(Level.SEVERE, null, ex);
        }

        m_types.clear();
        m_alphas.clear();
        m_potentialFunctions.clear();

        m_types.add(Boolean.FALSE);
        m_alphas.add(1.0);
        m_potentialFunctions.add(c);
    }

    /**
     * Construct instance from state feature vector
     *
     * @param stateFeature The state feature vector
     * @param weight The weight
     * @return
     */
    private Instance contructInstance2(double[] stateFeature, double weight) {
        int D = stateFeature.length;
        double[] values = new double[D + 1];
        System.arraycopy(stateFeature, 0, values, 0, D);
        DenseInstance ins = new DenseInstance(weight, values);
        return ins;
    }

    /**
     * Construct data head of instances (for the approximation in napping)
     *
     * @param D The number of feature vector
     * @param na The number of actions
     * @return The Instances
     */
    public Instances constructDataHead2(int D, int na) {
        ArrayList<Attribute> attInfo_x = new ArrayList<Attribute>();
        for (int i = 0; i < D; i++) {
            attInfo_x.add(new Attribute("att_" + i, i));
        }

        ArrayList<String> att = new ArrayList<String>(na);
        for (int i = 0; i < na; i++) {
            att.add("" + i);
        }
        attInfo_x.add(new Attribute("class", att, D));

        Instances data = new Instances("dataHead", attInfo_x, 0);
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    /**
     * Calculate the value of potential function
     *
     * @param ins The instance
     * @param A The number of actions
     * @param stateActionDim The dimension of state and action feature space
     * @param utilities The utility value to to be returned
     */
    private void calculatePV(Instance ins, int A, int stateActionDim, double[] utilities) {
        if (null == m_dataHead) {
            m_dataHead = constructDataHead(stateActionDim, A);
            m_nappingPool = constructDataHead(stateActionDim, A);
        }
        ins.setDataset(m_dataHead);

        double[] stateFeature = new double[stateActionDim - 1];
        for (int i = 0; i < stateFeature.length; i++) {
            stateFeature[i] = ins.value(i);
        }
        Instance ins2 = contructInstance2(stateFeature, 0);
        if (null == m_dataHead2) {
            m_dataHead2 = constructDataHead2(stateFeature.length, A);
        }
        ins2.setDataset(m_dataHead2);

        for (int j = 0; j < m_potentialFunctions.size(); j++) {
            if (m_types.get(j)) {
                try {
                    for (int a = 0; a < A; a++) {
                        ins.setValue(stateActionDim - 1, a);
                        utilities[a] += m_alphas.get(j) * m_potentialFunctions.get(j).classifyInstance(ins);
                    }
                } catch (Exception ex) {
                    Logger.getLogger(PolicyBoostDiscrete.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else {
                try {
                    double[] dist = m_potentialFunctions.get(j).distributionForInstance(ins2);
                    for (int a = 0; a < A; a++) {
                        utilities[a] += m_alphas.get(j) * dist[a];
                    }
                } catch (Exception ex) {
                    Logger.getLogger(PolicyBoostDiscrete.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
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
        for (int i = 0; i < numTraj; i++) {
            log_P_z[i] = 0;
            Trajectory trajectory = trainTrajectories.get(i);

            int T = trajectory.getSamples().size();
            for (int t = 0; t < T; t++) {
                Tuple tuple = trajectory.getSamples().get(t);
                double[] probabilities = getProbability(tuple.s, trajectory.getTask());
                log_P_z[i] += Math.log(probabilities[tuple.action.a]);
            }
        }

        for (int i = 0; i < numTraj; i++) {
            log_P_z[i] = log_P_z[i] / 1000;
        }
    }
}
