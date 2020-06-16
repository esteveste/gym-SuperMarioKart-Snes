# Super Mario Kart Snes - OpenAI Retro Integration 

![](assets/mario1.gif) 


This projects contains an integration of the Super Mario Kart to be used as a Gym environment.

We performed some initial reward shaping inspired on the **Sethbling** MarIQ code, that is explainned in a further section.
We also publish here some of our experiments that include several environment wrappers, preprocessing and agent code that allow a faster initial experimentation with the game. 

Youtube playlist with 4 agents trained on 4 different tracks - [https://www.youtube.com/playlist?list=PL7CojJo9fzQ1Qm7-n20kDOlE1u-A3DvzA](https://www.youtube.com/playlist?list=PL7CojJo9fzQ1Qm7-n20kDOlE1u-A3DvzA)


## Installation

### Dependencies

```bash
pip install gym-retro
```

### Installing the Game 

You need to copy the folder SuperMarioKart-Snes to your retro site-packages stable folder where the retro package is installed. 
Ending up with a path like **site-packages/retro/data/stable/SuperMarioKart-Snes**. 

If you are using Anaconda your retro data stable path should be something like:

```bash
#Linux
/home/user/miniconda3/envs/env_name/lib/python3.7/site-packages/retro/data/stable

#Windows
C:/Users/userName/Miniconda3/envs/env_name/Lib/site-packages/retro/data/stable
```

> Could be useful to make an symbolic link between the SuperMarioKart-Snes installed on the retro library and git folder to allow easier experimentation and modification

After this you will be able to import the game by doing

```python 
import retro
env = retro.make('SuperMarioKart-Snes')
env.reset()
for i in range(1000):
    obs,_,_,_ = env.step([1,0,0,0,0,0,0,0,0,0,0,0])
    env.render()
```
You can selected a certain state by doing (check the states in the folder)
```python
env = retro.make('SuperMarioKart-Snes','RainbowRoad_M')
```

You can also try to play the game by doing
```bash
python -m retro.examples.interactive --game SuperMarioKart-Snes
```


### Optional - Installing dependencies to run our Agent

Python requirements:
```bash
pip install --upgrade tensorflow==1.13.2
pip install stable-baselines[mpi]
```

If you are using a windows machine, you might need to install the Windows Message Passing Interface (MPI):
https://www.microsoft.com/en-us/download/details.aspx?id=57467



### Optional - Installing the OpenAI retro integration UI software

(OPTIONAL for configuring environment in retro and done/reward shaping )

Windows: https://storage.googleapis.com/gym-retro/builds/Gym%20Retro-latest-win64.zip

Mac: https://storage.googleapis.com/gym-retro/builds/Gym%20Retro-latest-Darwin.dmg

Extract the files and put them in the retro package main folder:
C:/Users/username/Miniconda3/envs/env_name/Lib/site-packages/retro

Then create a shortcut and run the .exe from there.



You can follow the guide : https://retro.readthedocs.io/en/latest/integration.html





## Training the Agent
You should ajust the settings in the main.py as you like and then just run it
```bash
python main.py
```

### Checking the learned model
To test the model you should make sure that the config is the same as the trained model, and then just load the file

```bash
python test_model_multiple.py --load best_model.zip
```

### Visualizing a run
After each run, a bk2 file should be created in the root directory.
To transform a run in a video file you can run 
```bash
python -m retro.scripts.playback_movie movies/<RunFile>.bk2
```

### Our Results
> To beat :)

| Track  |  Agent Best Time (min) |  
|---|---|
| MarioCircuit1  | 1:08:03  |  
| KoopaBeach1  | 1:13:49  |   
| GhostValley1  | 1:13:15  | 
| RainbowRoad | 1:50:27 |

If you get better results let us know!

## Future things to try
> Any contributions are greatly appreciated

The SuperMarioKart-Snes Game has the save stats for all tracks with the Mario on the Time Trial Setting. 
It could be interesting to also have save stats of different players and on the Mario GP setting.

It could also be interesting to integrate **Sethbling**'s idea of creating a reduced observation based on the track surface around the player. Allowing maybe a better agent generalization between tracks

If you know a better way to do experimentation and change multiple parameters in a single place and also keep track of the parameters used for each experiment, please let me know! 

## Contributors

* **esteveste** - Bernardo Esteves
* **CubeSkyy** - Miguel Coelho
* **Aegiel** - Francisco Lopes

---

Project for 2019-2020 Autonomous Agents and Multi-Agent Systems course in Instituto Superior Técnico

