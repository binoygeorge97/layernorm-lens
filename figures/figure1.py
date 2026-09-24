from pathlib import Path


if __name__ == "__main__":
    out = Path(__file__).with_name("figure1.txt")
    out.write_text("placeholder figure artifact\n")
