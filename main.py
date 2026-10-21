import argparse
import glob
import os

import acrobot as ab
from q_learning_agent import (ALPHA_START, EPS_START, create_q_table,
                              load_q_table, run, save_q_table)

MODEL_PATTERN = os.path.join("models", "Acrobot_v2_Q_table_*.pkl")


def episode_count_from_filename(filename):
    return int(os.path.splitext(os.path.basename(filename))[0].split("_")[-1])


def choose_model():
    files = sorted(glob.glob(MODEL_PATTERN), key=episode_count_from_filename)
    if not files:
        print("No Acrobot models found in the 'models' directory.")
        return None

    print("Available Models:")
    for i, f in enumerate(files, start=1):
        print(f"{i}. {os.path.basename(f)}")

    while True:
        try:
            choice = int(input("Select a model to load (enter the corresponding number): ")) - 1
            if choice < 0:
                raise IndexError
            return files[choice]
        except (IndexError, ValueError):
            print("Invalid choice. Please enter a number from the list.")


def train(episodes, model_file):
    if model_file:
        start = episode_count_from_filename(model_file)
        q = load_q_table(model_file)
        eps_start, alpha_start = 0.05, 0.1  # already trained: explore and overwrite less
        total = start + episodes
    else:
        q = None
        eps_start, alpha_start = EPS_START, ALPHA_START
        total = episodes

    env = ab.make_env(render=False)
    if q is None:
        q = create_q_table(env)
    run(env, q, episodes, learn=True, eps_start=eps_start, alpha_start=alpha_start)
    env.close()
    save_q_table(q, f"models/Acrobot_v2_Q_table_{total}.pkl")


def test(episodes, model_file, render):
    model_file = model_file or choose_model()
    if not model_file:
        return

    q = load_q_table(model_file)
    env = ab.make_env(render=render)
    run(env, q, episodes, learn=False)
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Acrobot Q-learning")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train a new model, or continue an existing one")
    p_train.add_argument("episodes", type=int)
    p_train.add_argument("model", nargs="?", help="existing model file to continue from")

    p_test = sub.add_parser("test", help="evaluate a trained model")
    p_test.add_argument("episodes", type=int)
    p_test.add_argument("model", nargs="?", help="model file (chosen from a list if omitted)")
    p_test.add_argument("--no-render", action="store_true",
                        help="no window: fast, for measuring the average score")

    args = parser.parse_args()
    if args.command == "train":
        train(args.episodes, args.model)
    elif args.command == "test":
        test(args.episodes, args.model, render=not args.no_render)


if __name__ == "__main__":
    main()