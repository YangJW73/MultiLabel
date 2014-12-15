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
import core.Action;
import java.io.Serializable;
import java.util.Random;

/**
 * The abstract class for task
 *
 * @author qing da <daq@lamda.nju.edu.cn>
 */
public abstract class Task implements Serializable {

    /**
     * The available actions
     */
    public Action[] actions;
    /**
     * The number of actions
     */
    public int actionDim;

    /**
     * Get the initial state
     *
     * @return
     */
    public abstract State getInitialState();

    /**
     * The transition method
     *
     * @param s The current state
     * @param a The action to be executed
     * @param outRand An instance of Random class
     * @return Next state
     */
    public abstract State transition(State s, Action a, Random outRand);

    /**
     * Return the immediate reward
     *
     * @param s The current state
     * @return The immediate reward
     */
    public abstract double immediateReward(State s);

    /**
     * Return true if the state is in the set of goal states
     *
     * @param s The current state
     * @return true if the state is in the set of goal states
     */
    public abstract boolean isComplete(State s);

    /**
     * Calculate the state and action feature vector
     *
     * @param s The state
     * @param action The action
     * @return The feature vector
     */
    public double[] getSAFeature(State s, Action action) {
        double[] feature = s.getfeatures();
        double[] saFea = new double[feature.length + 1];
        System.arraycopy(feature, 0, saFea, 0, feature.length);
        saFea[saFea.length - 1] = action.a;
        return saFea;
    }
}
