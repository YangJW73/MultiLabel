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

import core.Action;
import core.Trajectory;
import core.State;
import core.Task;
import java.util.List;
import java.util.Random;

/**
 * The random policy for discrete domain
 *
 * @author qing da <daq@lamda.nju.edu.cn>
 */
public class RandomD extends Policy {

    Random random;

    public RandomD(Random rand) {
        random = rand;
    }

    @Override
    public Action makeDecisionStochastic(State s, Task t, Random outRand) {
        Random thisRand = outRand == null ? random : outRand;
        int K = t.actions.length;

        double[] utilities = new double[K];
        double norm = 0;
        for (int k = 0; k < K; k++) {
            utilities[k] = 1;
            norm += utilities[k];
        }
        for (int k = 0; k < K; k++) {
            utilities[k] /= norm;
        }

        int bestAction = -1;
        double p = thisRand.nextDouble(), totalShare = 0;
        for (int k = 0; k < K; k++) {
            totalShare += utilities[k];
            if (p <= totalShare) {
                bestAction = k;
                break;
            }
        }

        return new Action(bestAction, utilities[bestAction]);
    }

    @Override
    public Action makeDecisionDeterministic(State s, Task t, Random outRand) {
        Random thisRand = outRand == null ? random : outRand;
        int K = t.actions.length;
        return new Action(thisRand.nextInt(K));
    }

    @Override
    public void update(List<Trajectory> rollouts) {
    }

    @Override
    public void setNumIteration(int numIteration) {
    }
}
