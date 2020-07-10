# Set Up Environment
from kaggle_environments import evaluate, make
from kaggle_environments.envs.halite.helpers import *


env = make("halite", configuration={ "episodeSteps": 400 }, debug=True)
print (env.configuration)

#print([action.name for action in ShipAction])
#state = env.state[0]
#board = Board(state.observation, env.configuration)
#print(board)

#env.run(["Submission.py", "random","random","random"])
#env.run(["Submission.py", "trainer_greedy.py", "random", "random"])
env.run(["Submission.py", "trainer_planned.py", "trainer_greedy.py", "trainer_swarm_int.py"])
env.render(mode="ipython", width=800, height=600)




