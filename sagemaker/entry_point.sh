
apt update
apt install libgl1-mesa-glx --no-install-recommends

pip install gym stable_baselines

TIMESTEPS=1e5 python train_climber.py

echo "Generated image $(ls ${SM_MODEL_DIR})"
