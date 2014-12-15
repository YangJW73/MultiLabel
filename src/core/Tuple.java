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
import core.State;

/**
 * The SARS tuple
 *
 * @author qing da <daq@lamda.nju.edu.cn>
 */
public class Tuple {

    public State s;
    public Action action;
    public double reward;
    public State sPrime;

    public Tuple(State s, Action a, double reward, State sPrime) {
        this.s = s;
        this.action = a;
        this.reward = reward;
        this.sPrime = sPrime;
    }
 
}
