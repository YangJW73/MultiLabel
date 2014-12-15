package domain;

import core.Action;
import core.EpsionGreedyExplorePolicy;
import core.Policy;
import core.Trajectory;
import core.Tuple;
import domain.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MLExecution {

	public static Trajectory runTaskWithFixedStep(MLTask task, MLState initalState, Policy policy, int maxStep, boolean isStochastic, Random random) {
        List<Tuple> samples = new ArrayList<Tuple>();
        ArrayList<Integer> predictOrder = new ArrayList<Integer>();

        MLState s;
        Action action;
        MLState sPrime = initalState;
        double reward = Double.NEGATIVE_INFINITY;

        int step = 0;
        double rewards = 0;
        while (step < maxStep && !task.isComplete(sPrime)) {
            s = sPrime;

            action = isStochastic ? policy.makeDecisionStochastic(s, task, random) : policy.makeDecisionDeterministic(s, task, random);
            
            if(predictOrder!=null){
            	while(predictOrder.contains(action.a))
                	action = isStochastic ? policy.makeDecisionStochastic(s, task, random) : policy.makeDecisionDeterministic(s, task, random);
                
                predictOrder.add(action.a);
            }
            else{
            	predictOrder.add(action.a);
            }
            
            sPrime = (MLState) task.transition(s, action, random);
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
