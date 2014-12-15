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
import core.Action;
import core.Policy;
import core.Trajectory;
import core.Tuple;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 *
 * @author qing da <daq@lamda.nju.edu.cn>
 */
public class Execution {

    /**
     * execute task for a single run and return the trajectoiry data
     * @param task The task
     * @param initalState The initial state
     * @param policy The policy
     * @param maxStep Maximal step for a single run of task 
     * @param isStochastic Whether using stochastic actions
     * @param random The instance of the Random class
     * @return 
     */
    public static Trajectory runTaskWithFixedStep(Task task, State initalState, Policy policy, int maxStep, boolean isStochastic, Random random) {
        List<Tuple> samples = new ArrayList<Tuple>();

        State s;
        Action action;
        State sPrime = initalState;
        double reward = Double.NEGATIVE_INFINITY;

        int step = 0;
        double rewards = 0;
        while (step < maxStep && !task.isComplete(sPrime)) {
            s = sPrime;

            action = isStochastic ? policy.makeDecisionStochastic(s, task, random) : policy.makeDecisionDeterministic(s, task, random);
            sPrime = task.transition(s, action, random);
            reward = task.immediateReward(sPrime);
            samples.add(new Tuple(s, action, reward, sPrime));

            rewards = rewards + reward;
            step = step + 1;
        }

        Trajectory rollout = new Trajectory(task, samples, maxStep, task.isComplete(sPrime));
        rollout.setRewards((rewards + reward * (maxStep - step)) / maxStep);
        return rollout;
    }
}
