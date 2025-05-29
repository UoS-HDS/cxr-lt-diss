from pathlib import Path


def main():
    print("Hello from dissertation!")
    print((Path(__file__).parent / ".venv").exists())


if __name__ == "__main__":
    main()
