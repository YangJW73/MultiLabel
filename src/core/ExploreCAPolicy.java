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
import java.util.List;
import java.util.Random;

/**
 *
 * @author qing da <daq@lamda.nju.edu.cn>
 */
public class ExploreCAPolicy extends Policy {

    /**
     * The random policy
     */
    private Policy rp;
    /**
     * Current policy
     */
    private Policy policy;

    /**
     * Construction method
     *
     * @param policy Current policy
     * @param random The instance of the Random class
     */
    public ExploreCAPolicy(Policy policy, Random random) {
        this.rp = new RandomC(new Random(random.nextInt()));
        this.policy = policy;
        this.m_random = random;
    }

    @Override
    public Action makeDecisionStochastic(State s, Task t, Random outRand) {
        Random thisRand = outRand == null ? m_random : outRand;
        Action policyAction = policy.makeDecisionStochastic(s, t, thisRand);
        if (policyAction == null) {
            policyAction = rp.makeDecisionStochastic(s, t, m_random);
        }
        return policyAction;
    }

    @Override
    public Action makeDecisionDeterministic(State s, Task t, Random outRand) {
        Random thisRand = outRand == null ? m_random : outRand;
        Action policyAction = policy.makeDecisionDeterministic(s, t, thisRand);
        if (policyAction == null) {
            policyAction = rp.makeDecisionStochastic(s, t, m_random);
        }
        return policyAction;
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
