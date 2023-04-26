from typing import Tuple


class Trainer:
    def __init__(self, model):
        self.model = model
        
    def _update(self, agent):
        if self.model == 'nn':
            advantages = agent.update_critic()
            agent.update_actor(advantages)
        else:
            for param in agent.policy.critic.parameters():
                param.requires_grad = True
                advantages = agent.update_critic()
                param.requires_grad = False
                
            for param in agent.policy.actor.parameters():
                param.requires_grad = True
                agent.update_actor(advantages)
                param.requires_grad = False
        agent.buffer.clear()

    def train(
        self,
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

                if t + 1 == max_steps:
                    done = True

                agent.buffer.rewards.append(reward)
                agent.buffer.terminals.append(done)
                cum_reward += reward

                if len(agent.buffer) == update_freq:
                    self._update(agent)

                if done:
                    break

                state = state_next
            returns.append(cum_reward)
            timesteps.append(t)
            print(f'{epoch}/{epochs}: {returns[-1]} \r', end='')
        return agent, returns, timesteps
