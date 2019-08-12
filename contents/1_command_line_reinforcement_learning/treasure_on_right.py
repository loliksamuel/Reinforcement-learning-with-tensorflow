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
    value_table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    # print(table)    # show table of curr state of weights what action is best. once the state is known, the   history may be thrown
    return value_table


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
        print(map_state2value)
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(SLEEP_TIME)


def choose_policy_action(state, q_table):
    # This is how to choose an action
    qualityOfCurrState = q_table.iloc[state, :]
    print(f'\nstate_actions={qualityOfCurrState.values}')
    if (np.random.uniform() > EPSILON) or ((qualityOfCurrState == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = qualityOfCurrState.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name


def get_env_feedback_value(S, A):
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




def rl():
    # main part of RL loop
    for episode in range(MAX_EPISODES):#agent's goal is to maximise the REWARD on given episode
        print (f'\nEpisode #{episode+1} ot of {MAX_EPISODES}')
        step_counter = 0
        agent_state = 0 # S= STATE, or current stair
        is_terminated = False
        print_env(agent_state, episode, step_counter)
        while not is_terminated:

            action = choose_policy_action(agent_state, map_state2value) #A=Action (left or right)
            print (f'chose action : {action}')
            agent_state_next, reward = get_env_feedback_value(agent_state, action)  # take action & get next state and reward
            print (f'q_reward {reward} . u r @ {agent_state_next} ')
            q_predict = map_state2value.loc[agent_state, action]
            print (f'q_predict {q_predict}')
            if agent_state_next != 'end---------------------------':
                q_target = reward + GAMMA * map_state2value.iloc[agent_state_next, :].max()   # next state is not end
            else:
                q_target = reward     # next state is end
                is_terminated = True    # terminate this episode
            print (f'q_target  {q_target}')
            update = ALPHA * (q_target - q_predict)
            print (f'q_update  {update}')
            map_state2value.loc[agent_state, action] +=  update # update
            agent_state = agent_state_next  #state is the agent's location which move to next state

            print_env(agent_state, episode, step_counter + 1)
            step_counter += 1
    return map_state2value



if __name__ == "__main__":

    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True) #prevent numpy exponential #notation on print, default False
    '''where are the
    1. algorithm model for value? this is a model free prediction and control (e.g. q-learning), it just use the agent_state
    1. algorithm model for policy? this is a model free rl, it just use the agent_state
    2. observations?
    3. policy map_state2action? is it value free rl? where nn : a3c, pg
    4. value?  is it policy free rl? where nn : ddqn , dqn
    5. exploration?
    6. exploitation?
    7. env state?
    8. agent state?
    9. why we need prediction here?
    10. where is planning?
    11. where is actor-critic?
    '''
    map_state2value = init_q_table(N_STATES, ACTIONS)
    print('\r\nQ-table initialized:\n')
    print(map_state2value)

    map_state2value = rl()
    print('\r\nQ-table end result:\n')
    print(map_state2value)
