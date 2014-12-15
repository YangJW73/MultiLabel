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
 *
 * @author qing da <daq@lamda.nju.edu.cn>
 */
public class Stat {

    /**
     * Calculate the mean and standard variance of x
     *
     * @param x
     * @return
     */
    public static double[] mean_std(double[] x) {
        double[] ms = new double[2];
        int N = x.length;

        ms[0] = 0;
        for (int i = 0; i < x.length; i++) {
            ms[0] += x[i];
        }
        ms[0] /= N;

        ms[1] = 0;
        for (int i = 0; i < x.length; i++) {
            ms[1] += (x[i] - ms[0]) * (x[i] - ms[0]);
        }
        ms[1] /= N;

        return ms;
    }
}
