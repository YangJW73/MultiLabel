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

import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * The class which implements the Runnable interface for parallel training of
 * the regreesion
 *
 * @author Yang Yu (yuy@nju.edu.cn)
 */
public class ParallelTrain implements Runnable {

    private Classifier c;
    private Instances data;

    public ParallelTrain(Classifier c, Instances data) {
        this.c = c;
        this.data = data;
    }

    @Override
    public void run() {
        try {
            c.buildClassifier(data);
        } catch (Exception ex) {
            Logger.getLogger(PolicyBoostDiscrete.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public Classifier getC() {
        return c;
    }
}