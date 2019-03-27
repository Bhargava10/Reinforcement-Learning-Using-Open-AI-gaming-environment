import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 4000


class DQNAgent:
    def __init__(self, state_size, action_size):
        # To see what Cartpole is learning, change it to "True"
        self.render = True

        # Used to generate the model by taking the size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # Hyper parameters of Cartpole DQN learning
        # Generate replay memory through deque
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_min = 0.005
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / 50000
        self.batch_size = 64
        self.train_start = 1000
        self.memory = deque(maxlen=10000)

        # Create a model to learn and a target model
        self.model = self.build_model()
        self.target_model = self.build_model()
        # Copy the model to be learned to the target model -> Initialize the target model (equalize and start the weight)
        self.update_target_model()

    # Approximate Q function through Deep Neural Network
    # state is input, and the Q value for each action is output.
    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # Update your target model to the model you are currently learning at regular time intervals
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # The choice of action uses the epsilon-greedy policy for the current network.
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # Save <s, a, r, s'> to replay_memory
    def replay_memory(self, state, action, reward, next_state, done):
        if action == 2:
            action = 1
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        # print(len(self.memory))

    # Random sampling of batch_size samples from replay memory
    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.action_size))

        for i in range(batch_size):
            state, action, reward, next_state, done = mini_batch[i]
            target = self.model.predict(state)[0]

            # Get the maximum Q Value at s' as in queuing. However, it is imported from the target model.
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.discount_factor * \
                                          np.amax(self.target_model.predict(next_state)[0])
            update_input[i] = state
            update_target[i] = target

        # Create a minibatch of the correct target answer and your own current value, and update the model at once
        self.model.fit(update_input, update_target, batch_size=batch_size, epochs=1, verbose=0)

    #Imported saved model
    def load_model(self, name):
        self.model.load_weights(name)

    # Saved learned model
    def save_model(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    # Up to 500 time steps for CartPole-v1
    env = gym.make('MountainCar-v0')
    # Bring state and behavior size from the environment
    state_size = env.observation_space.shape[0]
    #action_size = env.action_space.n
    action_size = 2
    # Creating a DQN Agent
    agent = DQNAgent(state_size, action_size)
    agent.load_model("H:\MountainCar_DQN.h5")
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        print(state)

        # Action 0 (left), 1 (do nothing), 3 (declare fake_action to avoid doing nothing
        fake_action = 0

        # Counter for the same action 4 times
        action_count = 0

        while not done:
            if agent.render:
                env.render()

            # Select an action in the current state and proceed to a step
            action_count = action_count + 1

            if action_count == 4:
                action = agent.get_action(state)
                action_count = 0

                if action == 0:
                    fake_action = 0
                elif action == 1:
                    fake_action = 2

            # Perform one step with the selected action
            next_state, reward, done, info = env.step(fake_action)
            next_state = np.reshape(next_state, [1, state_size])
            # Give a penalty of -100 for the action that ended the episode
            #reward = reward if not done else -100

            # Save <s, a, r, s'> to replay memory
            agent.replay_memory(state, fake_action, reward, next_state, done)
            #Study every time step
            agent.train_replay()
            score += reward
            state = next_state

            if done:
                env.reset()
                # Copy the learning model to each target episode
                agent.update_target_model()

                # For each episode, the time step where cartpole stood is plot
                scores.append(score)
                episodes.append(e)
                #pylab.plot(episodes, scores, 'b')
                #pylab.savefig("D:\MountainCar_DQN.h5")
                print("episode:", e, "  score:", score, "  memory length:", len(agent.memory),
                      "  epsilon:", agent.epsilon)

        # Save learning model for every 50 episodes
        if e % 50 == 0:
             agent.save_model("H:\MountainCar_DQN.h5")