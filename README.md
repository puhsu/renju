# Renju

## Main information
This is educational project aimed to understand and implement ideas and algorithms of deepmind
AlphaGo program, but in game called Gomoku (simpler variant of renju, also called five in a row).

This repository contains game enviroment with gui and implementation of different agents:
- Human 
- Dummy (makes random moves)
- Supervised Learning agent (uses policy network trained on expert moves dataset)
- BeamSearch agent (uses same policy network with beam search to improve it)
- Monte Carlo Tree Search agent (uses policy and rollout networks with mcts algorithm)

## Python dependencies
To install all requirements you can use virtualenv
```
virtualenv -p /path/to/python3.6.5 renju_env
renju_env/bin/activate
pip install -r requirements.txt
```
If you are using ubuntu you need to install tkinter 
```
sudo apt-get update
sudo apt-get install python3-tk
```

## Training from scratch
To train model from scratch you can use `./train` script. Run `./train --help` for details. If you then want to use
MCTS agent you need to convert keras model to tensorflow and freeze graph with `./keras2tf` script.

## Building MCTS agent
The code is tested on macOS 10.12 and Ubuntu 14.04. There are two ways of building this code:
1. You can use cmake to compile project and link dynamic tensorflow library (library built for macOS is in `src/libs/` (works faster).
```
cd src
wget https://www.dropbox.com/s/skszoznf0oih11f/libs.tar.gz
tar -xzf libs.tar.gz
rm libs.tar.gz
mkdir cmake-build && cd cmake-build
cmake .. -G"Unix Makefiles"
make bernard
```
2. If you don't want to use option 1. you can copy `src` folder inside `tensorflow` source directory and use bazel to build binary 
(this option will work slower for the first time)

```
git clone --recursive https://github.com/tensorflow/tensorflow /tmp/tensorflow
cp -r ./src /tmp/tensorflow/tensorflow
cd /tmp/tensorflow/
./configure # answer all the questions
cd tensorflow/src
bazel build :bernard # executable will be inside /tmp/tensorflow/bazel-bin/tensorflow/src/
```

## Demo 

To run demo simply use script `./demo --help`. This will launch python GUI with chosen
agents. You could also run test games between different agents (except human).
