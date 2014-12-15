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

/**
 * The Action class
 *
 * @author qing da <daq@lamda.nju.edu.cn>
 */
public class Action {

    /**
     * indicator for discrete actions
     */
    public int a;
    /**
     * control signals for continuous actions
     */
    public double[] controls;
    /**
     * probability of the action
     */
    public double probability;

    /**
     * set the control signals (for continuous actions)
     *
     * @param controls
     */
    public Action(double[] controls) {
        this.controls = controls;
        this.probability = 1;
    }

    /**
     * set the indicator of action (for discrete actions)
     *
     * @param a
     */
    public Action(int a) {
        this.a = a;
    }

    /**
     * set the indicator of action and corresponding probability
     *
     * @param a
     * @param probability
     */
    public Action(int a, double probability) {
        this(a);
        this.probability = probability;
    }

    /**
     * set the action and corresponding probability
     *
     * @param action
     * @param probability
     */
    public Action(Action action, double probability) {
        this.a = action.a;
        this.controls = action.controls;
        this.probability = probability;
    }

    /**
     * get the probability
     *
     * @return
     */
    public double getProbability() {
        return probability;
    }

    /**
     * set the probability
     *
     * @param probability
     */
    public void setProbability(double probability) {
        this.probability = probability;
    }
}
