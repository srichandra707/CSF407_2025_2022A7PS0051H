# Exercise 3: Tic-Tac-Toe & Wumpus World Integration

This project merges the **Tic-Tac-Toe** LLM implementations with a **Wumpus World** system. If LLM1 wins the tic-tac-toe matchup, it indicates a stronger agent; that agent then proceeds to act in the Wumpus World environment. The program also visualizes agent actions in the Wumpus World and provides risk probabilities in each cell.

---

## How to create conda env and add dependencies:

run this in the repo directory in the linux/macos terminal ; in windows, open conda terminal and run
```bash
conda env create -f config.yml
conda activate myenv
conda list
```

## How It Works

1. **Tic-Tac-Toe (LLM vs LLM)**  
   - The code first runs a series of **LLM vs LLM** tic-tac-toe trials (you specify the board size and number of trials).  
   - Both players use **Groq** LLMs (you will be prompted for API keys and model names).  
   - Results are saved in `Exercise1.json`, with outcome plots (`Exercise1_regular.png` and `Exercise1.png`).

2. **Wumpus World**  
   - After tic-tac-toe trials finish, you enter a **world size** for the Wumpus World.  
   - The code picks the winning agent (or fallback if a tie) and runs a Wumpus World simulation.  
   - The simulation uses Bayesian networks (`WumpusBN`) to display or compute risk probabilities.

---
