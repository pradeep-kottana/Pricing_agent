from dotenv import load_dotenv

load_dotenv()
import os


def main():
    print("Hello from pricing-agent!")
    print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))


if __name__ == "__main__":
    main()
