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
import core.State;
import core.EpsionGreedyExplorePolicy;
import core.ExploreCAPolicy;
import core.Policy;
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

/**
 *
 * @author qing da <daq@lamda.nju.edu.cn>
 */
public class Experiment {

    /**
     * the inner class which implements the Runnable interface for parallel
     * executions of tasks
     */
    class ParallelExecute implements Runnable {

        private Trajectory rollout;
        private Task task;
        private Policy policy;
        private State initialState;
        private int maxStep;
        private Random random;
        boolean isStochastic;

        public ParallelExecute(Task task, Policy policy, State initialState,
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
            rollout = Execution.runTaskWithFixedStep(task,
                    initialState, policy, maxStep, isStochastic, random);
        }

        public Trajectory getRollout() {
            return rollout;
        }
    }

    /**
     * conduct the experiment of iterative policy learning for domains with
     * discrete actions
     *
     * @param policy The policy
     * @param task The task
     * @param iteration Number of iterations of learning
     * @param trialsPerIter Number of task trials per iteration
     * @param initialState The initial state
     * @param maxStep Maximal step for a single run of task
     * @param isPara Whether using parallel programming
     * @param epsion The epsion probability for \epsion-greedy exploration
     * @param random The instance of the Random class
     * @return Some statistic information
     */
    public double[][] conductExperimentTrainDiscreteAction(Policy policy, Task task, int iteration, int trialsPerIter,
            State initialState, int maxStep, boolean isPara, double epsion, Random random) {

        double[][] results = new double[iteration][6];
        for (int iter = 0; iter < iteration; iter++) {
            results[iter][0] = iter;
            System.out.print("iter=" + iter + ", ");

            Policy explorePolicy = new EpsionGreedyExplorePolicy(policy, epsion, new Random(random.nextInt()));
            List<ParallelExecute> list = new ArrayList<ParallelExecute>();

            ExecutorService exec = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors() - 1);
            for (int i = 0; i < trialsPerIter; i++) {
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
            double[] rewards = new double[list.size()];
            double[] steps = new double[list.size()];
            int cc = 0;
            for (ParallelExecute run : list) {
                Trajectory rollout = run.getRollout();
                rollout.setProducedIteration(iter);
                rollouts.add(rollout);

                double totalReward = rollout.getRewards();
                rewards[cc] = totalReward;
                steps[cc] = rollout.getSamples().size();
                cc++;
            }
            double[] meanStdReward = Stat.mean_std(rewards);
            double[] meanStdStep = Stat.mean_std(steps);
            results[iter][1] = meanStdReward[0];
            results[iter][2] = meanStdReward[1];
            results[iter][3] = meanStdStep[0];
            results[iter][4] = meanStdStep[1];
            System.out.println("Average Total Rewards = " + String.format("%6f", meanStdReward[0]) + ", Average step = " + meanStdStep[0]);

            policy.update(rollouts);
        }

        return results;
    }

    /**
     * conduct the experiment of iterative policy learning for domains with
     * continuous actions
     *
     * @param policy The policy
     * @param task The task
     * @param iteration Number of iterations of learning
     * @param trialsPerIter Number of task trials per iteration
     * @param initialState The initial state
     * @param maxStep Maximal step for a single run of task
     * @param isPara Whether using parallel programming
     * @param epsion The epsion probability for \epsion-greedy exploration
     * @param random The instance of the Random class
     * @return Some statistic information
     */
    public double[][] conductExperimentTrainContinuousAction(Policy policy, Task task, int iteration, int trialsPerIter,
            State initialState, int maxStep, boolean isPara, Random random) {

        double[][] results = new double[iteration][5];
        for (int iter = 0; iter < iteration; iter++) {
            System.out.print("iter=" + iter + ", ");

            Policy explorePolicy = new ExploreCAPolicy(policy, new Random(random.nextInt()));
            List<ParallelExecute> list = new ArrayList<ParallelExecute>();

            ExecutorService exec = Executors.newFixedThreadPool(
                    Runtime.getRuntime().availableProcessors() - 1);
            for (int i = 0; i < (iter == 0 ? 1000 : trialsPerIter); i++) {
                ParallelExecute run = new ParallelExecute(task, explorePolicy, initialState, maxStep, i == 0 ? false : true, random.nextInt());
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
                    ex.printStackTrace();
                }
            }

            List<Trajectory> rollouts = new ArrayList<Trajectory>();
            double[] rewards = new double[list.size()];
            double[] steps = new double[list.size()];
            int cc = 0, maxStepUsed = -1;
            for (ParallelExecute run : list) {
                Trajectory rollout = run.getRollout();
                rollout.setProducedIteration(iter);
                rollouts.add(rollout);

                double totalReward = rollout.getRewards();
                rewards[cc] = totalReward;
                steps[cc] = rollout.getSamples().size();

                if (steps[cc] > maxStepUsed) {
                    maxStepUsed = rollout.getSamples().size();
                }

                cc++;
            }
            double[] meanStdReward = Stat.mean_std(rewards);
            double[] meanStdStep = Stat.mean_std(steps);
            results[iter][1] = meanStdReward[0];
            results[iter][2] = meanStdReward[1];
            results[iter][3] = meanStdStep[0];
            results[iter][4] = meanStdStep[1];
            System.out.println("Average Total Rewards = " + meanStdReward[0] + ", Average step = " + meanStdStep[0] + "(" + maxStepUsed + ")");

            policy.update(rollouts);
        }

        return results;
    }
}
