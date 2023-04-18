def REINFORCE(
        env,
        agent,
        epochs=100,
        max_steps=1000
):
    totals, timesteps = [], []
    for epoch in range(epochs):
        state, _ = env.reset()
        cum_reward = 0
        for t in range(max_steps):
            action = agent.select_action(state)
            state_next, reward, done, _, _ = env.step(action)

            if t + 1 == max_steps:
                done = True

            agent.buffer.rewards.append(reward)
            agent.buffer.terminals.append(done)
            cum_reward += reward

            if done:
                break

            state = state_next
        agent.update()

        totals.append(cum_reward)
        timesteps.append(t)
        print(f'{epoch}/{epochs}: {totals[-1]} - {t} \r', end='')

    return agent, totals, timesteps
