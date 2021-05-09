# Training-the-agent-Snake-using-Q-learning_AI_FinalProject

**Introduction:**

  Snake is a game in which a snake needs to explore an environment and catch the fruit without hitting any obstacle or itself. The agent gets rewarded everytime when it catches the fruit, also the size of snake increases. At first, the snake doesn’t know how to eat the food and is less “purposeful”. It also tends to die a lot by going the opposite way that its currently going and immediately hitting its tail. But it doesn’t take very long for the agent to learn how to play the game. The agent snake will play the game in a 16x16 grid.
  
**Algorithm used:**
   
  ***Q-learning Algorithm:***
  
  Q-learning is a reinforcement learning method that teaches a learning agent how to perform a task by rewarding good behavior and punishing bad behavior. In Snake, for example, moving closer to the food is good.
  
  In Q-learning algorithm, first the Q-table is initialized with all zeros. Then the action is chosen and performed, from there the rewards are measured. Finally, the Q-table is updated using the Bellman equation.
  
Bellman Equation:
 Q[state, action] = reward + gamma * max(Q[new_state, :])

**Instructions to run the code:**

Open the snake_ai.py file

Run the file
