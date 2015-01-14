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

import core.State;
import core.Task;
import core.Action;

import java.util.ArrayList;
import java.util.Random;

/**
 *
 * @author qing da <daq@lamda.nju.edu.cn>
 */
public abstract class GibbsPolicy extends Policy {

    /**
     * Given the state and task, calcualte the utility value for each action
     *
     * @param s The state
     * @param t The task
     * @return The utility value for each action
     */
    public abstract double[] getUtility(State s, Task t);

    @Override
    public Action makeDecisionDeterministic(State s, Task t, Random outRand) {
        if (m_numIteration == 0) {
            return null;
        }

        Random thisRand = outRand == null ? m_random : outRand;
        int K = t.actions.length;

        double[] probabilities = getProbability(s, t);
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

    /**
     * Given the state, task and the probabilities of each action, making a
     * stochastic action according to the probabilities
     *
     * @param s
     * @param t
     * @param probabilities
     * @param outRand
     * @return
     */
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
    
    /*
     * written by yjw
     * make decision within available action set
     */
    public Action makeDecisionStochastic(State s, Task t, double[] probabilities, ArrayList<Integer> usedAction, Random outRand) {
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

    /**
     * Calculate the probabilities of each action
     *
     * @param s The state
     * @param t The task
     * @return The probabilities of each action
     */
    public double[] getProbability(State s, Task t) {
        double[] utilities = getUtility(s, t);
        return getProbability(utilities);
    }

    /**
     * Calculate the getProbability according to the utility value
     *
     * @param utilities
     * @return
     */
    public double[] getProbability(double[] utilities) {
        double[] probabilities = new double[utilities.length];
        double maxUtility = Double.NEGATIVE_INFINITY;
        for (int k = 0; k < utilities.length; k++) {
            if (utilities[k] > maxUtility) {
                maxUtility = utilities[k];
            }
        }

        double norm = 0;
        for (int k = 0; k < utilities.length; k++) {
            probabilities[k] = Math.exp(utilities[k] - maxUtility + 10);
            norm += probabilities[k];
        }


        for (int k = 0; k < probabilities.length; k++) {
            probabilities[k] /= norm;
        }

        if (m_numIteration == 160) {
            for (int i = 0; i < probabilities.length; i++) {
                System.err.print(probabilities[i] + ",");
            }
            System.err.println();
        }

        return probabilities;
    }

	
}
