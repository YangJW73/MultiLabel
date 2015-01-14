package domain;

import core.EpsionGreedyExplorePolicy;
import core.Policy;
import core.Task;
import core.State;
import core.PolicyBoostDiscrete;
import core.Stat;
import core.Trajectory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

public class MLExperiment {

	public Trajectory bestTraj;
    /**
     * the inner class which implements the Runnable interface for parallel
     * executions of tasks
     */
    class ParallelExecute implements Runnable {

        private Trajectory rollout;
        private MLTask task;
        private Policy policy;
        private MLState initialState;
        private int maxStep;
        private Random random;
        boolean isStochastic;

        public ParallelExecute(MLTask task, Policy policy, MLState initialState,
                int maxStep, boolean isStochastic, int seed) {
            this.task = task;
            this.policy = policy;
            this.initialState = initialState;
            this.maxStep = maxStep;
            this.isStochastic = isStochastic;
            this.random = new Random(seed);
        }

        @Override
        public void run() {
            rollout = MLExecution.runTaskWithFixedStep(task,
                    initialState, policy, maxStep, isStochastic, random);
        }

        public Trajectory getRollout() {
            return rollout;
        }
    }

    public void trainPolicy(PolicyBoostDiscrete policy, MLTask task, int iteration, int trialsPerIter,
            MLState initialState, int maxStep, boolean isPara, double epsion, Random random) {

    	double maxReward = Double.NEGATIVE_INFINITY;
    	
    	for (int iter = 0; iter < iteration; iter++) {System.out.println("processing iter "+iter);
    		Policy explorePolicy = new EpsionGreedyExplorePolicy(policy, epsion, new Random(random.nextInt()));
    		List<ParallelExecute> list = new ArrayList<ParallelExecute>();

            ExecutorService exec = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors() - 1);
            for (int i = 0; i < trialsPerIter; i++) {System.out.println("processing trial "+i);
                ParallelExecute run = new ParallelExecute(task, explorePolicy, initialState, maxStep, true, random.nextInt());
                list.add(run);
                if (isPara && iter > 0) {
                    exec.execute(run);
                } else {
                    run.run();
                }
            }
            if (isPara && iter > 0) {
                exec.shutdown();
                try {
                    while (!exec.awaitTermination(10, TimeUnit.SECONDS)) {
                    }
                } catch (InterruptedException ex) {
                    Logger.getLogger(PolicyBoostDiscrete.class.getName()).log(Level.SEVERE, null, ex);
                }
            }

            List<Trajectory> rollouts = new ArrayList<Trajectory>();
            for (ParallelExecute run : list) {
                Trajectory rollout = run.getRollout();
                //rollout.setProducedIteration(iter);
                rollouts.add(rollout);

                double totalReward = rollout.getRewards();
                if(Double.compare(totalReward, maxReward)>0){
                	maxReward = totalReward;
                	bestTraj = rollout;
                }
            }
            policy.update(rollouts);
    	}
    }
}
