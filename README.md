# Lunar Climber

Machine Learning project for personal experimentation and learning.

Based on the Stable Baselines Lunar Lander example: https://stable-baselines.readthedocs.io/en/master/guide/examples.html

![Inspiration for the project](https://cdn-images-1.medium.com/max/960/1*f4VZPKOI0PYNWiwt0la0Rg.gif)

### Relevant files

The relevant files are mainly `train_climber.py`, `run_climber`, and `custom_end/lunar_climber_env.py`. The rest can be ignored relatively safely, including folders.

### GPU

Dockerfile.gpu is my attempt at creating a docker image that enables the use of GPU. I'm not sure if this actually works, as I can't test on my machine (no GPU) and I don't know what drivers exist on the machines running Sagemaker Notebook. I'm currently tes

I'm also not sure whether Tensorflow actually makes use of the GPU, given that I configure the Docker image correctly. According to [https://www.tensorflow.org/install/gpu](Tensorflow's GPU guide), any version _after_ tensorflow==1.15 combines CPU and GPU compilations into the same package. I am using exactly 1.15.0, however, so I suppose I need `pip install tensorflow-gpu`. I haven't been able to make this work yet.

[https://github.com/hill-a/stable-baselines/issues/308#issuecomment-546677348](This) Github issue comment seems to indicate that simply installing `tensorflow-gpu` instead of `tensorflow` will make the algorithm use GPU instead, but I'm not sure whether that's reliable.

Multi-processing is enabled simply by replacing

```
env = LunarLander()
```

with

```
env = SubprocVecEnv([lambda: LunarLander() for i in range(n_envs)])
```

The PPO2 algorithm I use automatically coordinates the interaction between the different processes. I assume this also works for GPU when I get that working.
