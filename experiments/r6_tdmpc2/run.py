from pathlib import Path


def main():
    cfg = Path(__file__).with_name("config.yaml")
    print(f"Running experiment with {cfg}")


if __name__ == "__main__":
    main()
