import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from env import HaxballEnv

MODEL_PATH = "models/ppo_haxball_final.zip"

N_STACK = 4


def make_env():
    def _init():
        return HaxballEnv()
    return _init


def main():

    env = DummyVecEnv([make_env()])

    env = VecFrameStack(env, n_stack=N_STACK)

    model = PPO.load(MODEL_PATH, env=env)

    obs = env.reset()

    while True:

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, done, info = env.step(action)

        if done[0]:
            print("Episode finished.")
            time.sleep(2)


if __name__ == "__main__":
    main()