"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
import time

np.random.seed(2)  # reproducible

ACTIONS       = ['left', 'right']     # available actions
MAX_EPISODES = 8   # maximum episodes (like epochs).  1 episode = one a sequence of states, actions and rewards, which ends with terminal state. for ex.  , playing an entire game can be considered as 1 episode
N_STATES     = 6    # the length of the 1 dimensional world has 6 stairs
EPSILON      = 0.9  # greedy police
ALPHA        = 0.1  # learning rate
GAMMA        = 0.9  # discount factor
SLEEP_TIME   = 0.3  # fresh time for one move


def init_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    # print(table)    # show table
    return table


def print_env(S, episode, step_counter):
    # This is how environment be updated
    print('\n')
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    #print (env_list)
    if S == 'end---------------------------':
        interaction = '------------------Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r    =================     got the end   ===============================                            ', end='')
        print('\r\nQ-table curr result:\n')
        print(q_table)
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(SLEEP_TIME)


def choose_action(state, q_table):
    # This is how to choose an action
    qualityOfCurrState = q_table.iloc[state, :]
    print(f'\nstate_actions={qualityOfCurrState.values}')
    if (np.random.uniform() > EPSILON) or ((qualityOfCurrState == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = qualityOfCurrState.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name


def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 'right':    # move right
        if S == N_STATES - 2:   # terminate
            S_ = 'end---------------------------'
            REWARD = 1# REWARD=1 instead of labeling data in RL we use feedback reward and it  sometimes  delayed
        else:
            S_ = S + 1
            REWARD = 0# REWARD=0
    else:   # move left
        REWARD = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, REWARD




def rl_agent():
    # main part of RL loop
    for episode in range(MAX_EPISODES):#agent's goal is to maximise the REWARD on given episode
        print (f'\nEpisode #{episode+1} ot of {MAX_EPISODES}')
        step_counter = 0
        step = 0 # S= STATE, or current stair
        is_terminated = False
        print_env(step, episode, step_counter)
        while not is_terminated:

            action = choose_action(step, q_table) #A=Action (left or right)
            print (f'chose action : {action}')
            step_, reward = get_env_feedback(step, action)  # take action & get next state and reward
            print (f'q_reward {reward} . u r @ {step_} ')
            q_predict = q_table.loc[step, action]
            print (f'q_predict {q_predict}')
            if step_ != 'end---------------------------':
                q_target = reward + GAMMA * q_table.iloc[step_, :].max()   # next state is not end
            else:
                q_target = reward     # next state is end
                is_terminated = True    # terminate this episode
            print (f'q_target  {q_target}')
            update = ALPHA * (q_target - q_predict)
            print (f'q_update  {update}')
            q_table.loc[step, action] +=  update # update
            step = step_  # move to next state

            print_env(step, episode, step_counter + 1)
            step_counter += 1
    return q_table



if __name__ == "__main__":

    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True) #prevent numpy exponential #notation on print, default False


    q_table = init_q_table(N_STATES, ACTIONS)
    print('\r\nQ-table initialized:\n')
    print(q_table)

    q_table = rl_agent()
    print('\r\nQ-table end result:\n')
    print(q_table)
