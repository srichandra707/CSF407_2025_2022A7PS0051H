# Exercise 3: Tic-Tac-Toe & Wumpus World Integration

This project merges the **Tic-Tac-Toe** game (using Groq LLMs) with a **Wumpus World** simulation. The outcome of the Tic-Tac-Toe trials decides which agent (LLM1 or LLM2) will operate in the Wumpus World:
- If **LLM1** wins at tic-tac-toe, it uses a **best-move approach** in the Wumpus World.
- If **LLM2** wins at tic-tac-toe, it uses a **random-move approach** in the Wumpus World.

After the trials, you specify a Wumpus World size, and the chosen agent strategy is applied to that world.

---

## How to create conda env and add dependencies:

run this in the repo directory in the linux/macos terminal ; in windows, open conda terminal and run
```bash
conda env create -f config.yml
conda activate myenv
conda list
```
## How to run the rogram:
- Go to `src` folder -> open `Exercise3.py` -> run it
---
## Working of the System

1. **Tic-Tac-Toe Trials**  
   - You specify a board size (e.g., `3` for a 3x3 board) and the number of trials.
   - Each trial features **LLM1** (Groq) vs **LLM2** (Groq).
   - **LLM1** attempts to find a winning, blocking, or setup move using heuristics (and possibly LLM input).
   - **LLM2** uses random moves if it wins the game, demonstrating a contrasting approach.

2. **Results & Visualization**  
   - Outcomes (LLM1 win, LLM2 win, draw) are recorded in `Exercise1.json`.
   - Two plots are generated:
     - `Exercise1_regular.png`: A bar chart of the outcomes.
     - `Exercise1.png`: A binomial distribution plot showing the probability of LLM1 wins.

3. **Wumpus World**  
   - After the tic-tac-toe trials, you enter a “world size.”
   - The system initializes an **EnhancedWumpusWorld** with an agent strategy based on which LLM performed better.
   - The simulation runs, and you can observe the agent’s risk-based actions in the environment.

---

## Conclusion

This setup demonstrates how two distinct LLM approaches, best-move vs. random, perform and how the winner transitions to solving a Wumpus World scenario. 

