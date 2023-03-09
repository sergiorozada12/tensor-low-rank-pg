from typing import Tuple


def TRPO(
    env,
    agent,
    epochs: int,
    max_steps: int,
    update_freq: int
):
    returns = []
    timesteps = []
    for epoch in range(epochs):
        state, _ = env.reset()
        cum_reward = 0

        for t in range(max_steps):
            action = agent.select_action(state)
            state_next, reward, done, _, _ = env.step(action)

            agent.buffer.rewards.append(reward)
            agent.buffer.terminals.append(done)
            cum_reward += reward

            if len(agent.buffer) == update_freq:
                agent.update()

            if done:
                break

            state = state_next
        returns.append(cum_reward)
        timesteps.append(t)
        print(f'{epoch}/{epochs}: {returns[-1]} \r', end='')
    return agent, returns, timesteps
