# Tictactoe Using LLMs

This project implements a **Tic-Tac-Toe** game in Python, leveraging **Groq** Large Language Models (LLMs). You can run two different LLMs against each other (LLM1 vs LLM2) or play against one of the LLMs yourself.

---
## How to run the program:

- Go to `src` folder -> open `tictactoe.py` -> run it -> enter the following values: 

## How It Works

1. **API Keys**: You can provide your **Groq** API keys either via a `.env` file (optional) or directly on the CLI prompts.
2. **Multiple Models**: Each LLM (LLM1 and LLM2) can be assigned a different Groq model.
3. **Board Size**: Choose any NxN board size (e.g., 3 for a 3x3 game).
4. **Game Modes**:  
   - **Mode 1**: LLM1 vs LLM2  
     - You will be asked for the **number of trials**. The game automatically runs that many times and collects stats.  
   - **Mode 2**: LLM1 vs Human  
     - You (the human) enter moves by specifying row/col coordinates.

---
## Prompt and Response

The program constructs a detailed prompt that informs the LLM about:
- Your player number and symbol.
- The current board state (formatted as a string).
- The opponent’s last move.
- All available empty spots.
- Prioritized instructions: win immediately, block the opponent, or set up a win.
- A requirement to return the move exactly as a coordinate pair in the format `(row,col)` with no extra text.

This prompt is sent using the Groq client's `chat.completions.create` method, and the returned response (from `response.last`) is parsed to determine the next move.

![img1](https://github.com/srichandra707/CSF407_2025_2022A7PS0051H/blob/15dc925a30bb15df9d6a9829d0c2c6b7d238d523/Excercise1/src/Screenshot%202025-03-02%20152053.png)

---

## Visualization of outcomes:

### Based on the number of trials a binomial distribution is plotted, for example number of trials = 50,

![img2](https://github.com/srichandra707/CSF407_2025_2022A7PS0051H/blob/241f959bc8694a6c47a292cfc70bcbeedd126821/Excercise1/src/Exercise1.png
)


### A bar graph has also been plotted basing on the outcomes of various trials

![img3](https://github.com/srichandra707/CSF407_2025_2022A7PS0051H/blob/15dc925a30bb15df9d6a9829d0c2c6b7d238d523/Excercise1/src/Exercise1_regular.png)

---

## For Human vs LLM1
In this mode, the game displays the current board and prompts the human for a move (in "row,col" format) when it's their turn; otherwise, the LLM generates its move using the configured API. The game alternates moves until a win or draw is detected, then prints the final board state and outcome.

![img4](https://github.com/srichandra707/CSF407_2025_2022A7PS0051H/blob/15dc925a30bb15df9d6a9829d0c2c6b7d238d523/Excercise1/src/Screenshot%202025-03-02%20152557.png)

---
## Conclusion

Our Tic-Tac-Toe game seamlessly integrates **LLMs** via Groq APIs for dynamic move generation in both automated **(LLM vs LLM)** and interactive **(Human vs LLM)** modes. This demonstrates the practical application of language models to enhance classic games.


