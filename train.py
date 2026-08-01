import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from env import HaxballEnv

MODEL_DIR = "models"
TOTAL_TIMESTEPS = 200_000

N_STACK = 4


def make_env():
    def _init():
        return HaxballEnv()
    return _init


def main():

    os.makedirs(MODEL_DIR, exist_ok=True)

    env = DummyVecEnv([make_env()])
    env = VecFrameStack(env, n_stack=N_STACK)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=MODEL_DIR,
        name_prefix="ppo_haxball_ckpt"
    )
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="./tb_logs"
    )

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback
        )
    except KeyboardInterrupt:
        print("Training interrupted, saving current model...")
    finally:
        model.save(os.path.join(MODEL_DIR, "ppo_haxball_final"))
        env.save(os.path.join(MODEL_DIR, "vecnormalize.pkl"))
        env.close()


if __name__ == "__main__":
    main()