import numpy as np
import time
import scipy

def sample_trajectory(env,policy,max_path_length):
    ob = env.reset()
    
    obs, acs, rewards, next_obs, terminals = [], [], [], [], []
    steps = 0
    
    while True:
        
        obs.append(ob)
        ac = policy.get_action(ob) 
        ac = ac[0]
        acs.append(ac[0])
        
        ob, rew, done, _ = env.step(ac)
        
        step += 1
        next_obs.append(ob)
        rewards.append(rew)
        
        rollout_done =(done or (steps == max_path_length)) # HINT: this is either 0 or 1, done is typically determined by the env.step
        terminals.append(rollout_done)
        
        if rollout_done:
            break
            
    return Path(obs,acs,rewards,next_obs,terminals)
    

def Path(obs,acs,rewards,next_obs,terminals):
    """
        Take info (separate Lists) from a single rollout
        and return it in a single dictionary
    """
    return {"observation" : np.array(obs, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "reward" : np.array(rewards, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}
            
def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length):
    # min_timesteps_per_batch , I need to sample this number of steps

    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        paths.append(sample_trajectory(env, policy, max_path_length))
        timesteps_this_batch += get_pathlength(paths[-1])

    return paths, timesteps_this_batch
    
def get_pathlength(path):

    return len(path["reward"])
    

def sample_n_trajectories(env, policy, ntraj, max_path_length):
    """
        Collect ntraj rollouts.
    """
    paths = []
    for n in range(ntraj):
        paths.append(sample_trajectory(env, policy, max_path_length))

    return paths
        

def convert_listofrollouts(paths, concat_rew=True):       
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    if concat_rew:
        rewards = np.concatenate([path["reward"] for path in paths])
    else:
        rewards = [path["reward"] for path in paths]
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    return observations, actions, rewards, next_observations, terminals
    # the output is numpy array