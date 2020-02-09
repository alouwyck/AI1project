# OpenAI Gym importeren
import gym

# FrozenLake environment aanmaken
env = gym.make("FrozenLake-v0")

# aantal actions en states
print("Action space: ", env.action_space)
print("Observation space: ", env.observation_space)

# environment resetten
env.reset()
env.render()

# speel het spel...
maxiter = 10
for i in range(maxiter):
    random_action = env.action_space.sample()
    new_state, reward, done, info = env.step(random_action)
    env.render()
    if done:
        break
