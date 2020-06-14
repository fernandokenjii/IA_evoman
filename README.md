# IA_evoman
Project developed to the WCCI competition, first introduced on UFABC's Artificial Intelligence class.

## Files

* [our_controller.py](our_controller.py): Creates the Artificial Neural Network, preprocess inputs and sends the agent's actions to the game.
* [our_optimization.py](our_optimization.py): Defines the environment and the Genetic Algorithm.

The outputs are presented in the `tests` folder, containing:
* [evoman_logs.txt](our_tests/evoman_logs.txt): logs from evoman generated during execution
* [results.txt](our_tests/results.txt): logs generations and their best fitness
* [Evoman.pkl](our_tests/Evoman.pkl): all ANN weights
* [results.csv](results.csv): the results for each iteration containing:
  - each train boss fight result and their average
  - each test boss fight result and their average
  - global average
  - gain (harmonic mean)
  - our custom fitness
* [results.png](results.png): the graphic between the gain and the fitness function on each iteration

## Environment set-up

Make sure you have python 3.6 or 3.7, pip and venv dependencies installed.
If not, you can install them using:

**Ubuntu/Debian:**

```bash
sudo apt-get install python3
sudo apt-get install python3-pip
sudo apt-get install python3-venv
```

Then install the dependencies:
```bash
# Create a new virtual environment
python3 -m venv env

# Activate the environment
source env/bin/activate

# Upgrade pip version
pip install --upgrade pip

# Install all required dependencies
pip install -r requirements.txt
```

## Running

Make sure that venv is active, then to run the program, execute `python our_optimization.py`

You can change between the train and test modes by editing the *mode* variable to *'train'* or *'test'*.
It's also possible to select the enemies for the training phase editing the *enemies* variable in the same file.

On train mode, the graphic interface is deactivated and, at the end of iterations (defined on _GA_ class), it runs a final test with the best agent.

## Results and Conclusions

We present our results and conclusions in [this paper](Evoman_WCCI_Competition.pdf).
