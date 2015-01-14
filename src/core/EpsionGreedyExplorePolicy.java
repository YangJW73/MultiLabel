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
import core.Trajectory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 *
 * @author qing da <daq@lamda.nju.edu.cn>
 */
public class EpsionGreedyExplorePolicy extends Policy {

    /**
     * The random policy
     */
    private Policy rp;
    /**
     * Current policy
     */
    private Policy policy;
    /**
     * The epsion probability for \epsion-greedy exploration
     */
    private double epsion;

    /**
     * Construction method
     *
     * @param policy The current policy
     * @param epsion The epsion probability for \epsion-greedy exploration
     * @param random The instance of the Random class
     */
    public EpsionGreedyExplorePolicy(Policy policy, double epsion, Random random) {
        this.rp = new RandomD(new Random(random.nextInt()));
        this.policy = policy;
        this.epsion = epsion;
        this.m_random = random;
    }

    @Override
    public Action makeDecisionStochastic(State s, Task t, Random outRand) {
        Random thisRand = outRand == null ? m_random : outRand;
        double[] probabilities = ((GibbsPolicy) policy).getProbability(s, t);
        Action policyAction = ((GibbsPolicy) policy).makeDecisionStochastic(s, t, probabilities, thisRand);

        Action action;
        if (thisRand.nextDouble() < epsion || policyAction == null) {
            action = new Action(rp.makeDecisionDeterministic(s, t, thisRand), -1);
            action.setProbability(epsion);
        } else {
            action = new Action(policyAction, -1);
        }

        if (policyAction == null) {
            action.setProbability(1.0 / t.actions.length);
        } else {
            action.setProbability(epsion / t.actions.length + (1 - epsion) * probabilities[action.a]);
        }
        return action;
    }
    
    @Override
    public Action makeDecisionStochastic(State s, Task t, ArrayList<Integer> usedAction, Random outRand) {
        Random thisRand = outRand == null ? m_random : outRand;
        double[] probabilities = ((GibbsPolicy) policy).getProbability(s, t);
        Action policyAction = ((GibbsPolicy) policy).makeDecisionStochastic(s, t, probabilities,usedAction, thisRand);

        Action action;
        if (thisRand.nextDouble() < epsion || policyAction == null) {
            action = new Action(rp.makeDecisionDeterministic(s, t, thisRand), -1);
            action.setProbability(epsion);
        } else {
            action = new Action(policyAction, -1);
        }

        if (policyAction == null) {
            action.setProbability(1.0 / t.actions.length);
        } else {
            action.setProbability(epsion / t.actions.length + (1 - epsion) * probabilities[action.a]);
        }
        return action;
    }

    @Override
    public Action makeDecisionDeterministic(State s, Task t, Random outRand) {
        Random thisRand = outRand == null ? m_random : outRand;
        Action policyAction = policy.makeDecisionDeterministic(s, t, thisRand);

        Action action;
        if (thisRand.nextDouble() < epsion || policyAction == null) {
            action = new Action(rp.makeDecisionDeterministic(s, t, thisRand), -1);
            action.setProbability(epsion);
        } else {
            action = new Action(policyAction, -1);
        }

        if (policyAction == null) {
            action.setProbability(1.0 / t.actions.length);
        } else if (action.a == policyAction.a) {
            action.setProbability(epsion / t.actions.length + (1 - epsion));
        } else {
            action.setProbability(epsion / t.actions.length);
        }
        return action;
    }

    @Override
    public void update(List<Trajectory> rollouts) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public void setNumIteration(int numIteration) {
        throw new UnsupportedOperationException("Not supported yet.");
    }
}
