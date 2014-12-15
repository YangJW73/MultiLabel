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
import java.io.Serializable;
import java.util.List;
import java.util.Random;

/**
 *
 * @author qing da <daq@lamda.nju.edu.cn>
 * @author Yang Yu <eyounx@gmail.com>
 */
public abstract class Policy implements Serializable {

    protected int m_numIteration;
    protected Random m_random;

    /** 
     * Make a decision with probability
     * @param s current state
     * @param t current task
     * @param random
     * @return action in <code>Action</code>
     */
    public abstract Action makeDecisionStochastic(State s, Task t, Random random);

    /** 
     * Make a decision with probability 1
     * @param s current state
     * @param t current task
     * @param random
     * @return action in <code>Action</code>
     */
    public abstract Action makeDecisionDeterministic(State s, Task t, Random random);

    /**
     * update the policy from the roll-outs
     * @param rollouts set
     */
    public abstract void update(List<Trajectory> rollouts);

    /**
     * current number of iterations of policy improvement
     * @return number in <code>int</code>
     */
    public int getNumIteration() {
        return m_numIteration;
    }

    /**
     * number of iterations of policy improvement
     * @param n number in <code>int</code>
     */
    public void setNumIteration(int n){
        m_numIteration = n;
    }

    public Random getRandom() {
        return m_random;
    }
}
