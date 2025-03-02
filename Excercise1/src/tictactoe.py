import json
import random
import os
import re
import time
import matplotlib.pyplot as plt
import numpy as np
from groq import Groq
from scipy.stats import binom
from dotenv import load_dotenv

load_dotenv()
print("Loaded .env file from:", os.path.abspath(".env") if os.path.exists(".env") else "Not found")
print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY")[:5] + "..." if os.getenv("GROQ_API_KEY") else "pls set the key")
print("GROQ_API_KEY1:", os.getenv("GROQ_API_KEY1")[:5] + "..." if os.getenv("GROQ_API_KEY1") else "pls set the key")

class TicTacToe:
    def __init__(self, n):
        self.n = n
        self.board = [[0] * n for _ in range(n)]
        self.current_player = 1

    def make_move(self, row, col):
        if 0 <= row < self.n and 0 <= col < self.n and self.board[row][col] == 0:
            self.board[row][col] = self.current_player
            self.current_player = 3 - self.current_player
            return True
        print(f"Invalid move: ({row}, {col})")
        return False

    def check_win(self):
        for row in self.board:
            if row.count(row[0]) == self.n and row[0] != 0:
                return row[0]
        for col in range(self.n):
            if all(self.board[row][col] == self.board[0][col] != 0 for row in range(self.n)):
                return self.board[0][col]
        if all(self.board[i][i] == self.board[0][0] != 0 for i in range(self.n)):
            return self.board[0][0]
        if all(self.board[i][self.n-1-i] == self.board[0][self.n-1] != 0 for i in range(self.n)):
            return self.board[0][self.n-1]
        if all(self.board[row][col] != 0 for row in range(self.n) for col in range(self.n)):
            return -1  # Draw
        return 0

    def board_to_string(self):
        return "\n".join([" ".join(map(lambda x: 'X' if x == 1 else 'O' if x == 2 else '.', row)) for row in self.board])

    def get_empty_spots(self):
        return [(r, c) for r in range(self.n) for c in range(self.n) if self.board[r][c] == 0]

def parse_move(response):
    match = re.search(r'\((\d+),(\d+)\)', response)
    if match:
        row, col = int(match.group(1)), int(match.group(2))
        return (row, col)
    return None

def find_winning_move(game, player):
    empty_spots = game.get_empty_spots()
    for move in empty_spots:
        temp_game = TicTacToe(game.n)
        temp_game.board = [row[:] for row in game.board]
        temp_game.current_player = player
        temp_game.make_move(*move)
        if temp_game.check_win() == player:
            return move
    return None

def find_blocking_move(game, player):
    opponent = 3 - player
    return find_winning_move(game, opponent)

def find_setup_move(game, player):
    empty_spots = game.get_empty_spots()
    for move in empty_spots:
        temp_game = TicTacToe(game.n)
        temp_game.board = [row[:] for row in game.board]
        temp_game.current_player = player
        temp_game.make_move(*move)
        next_empty = temp_game.get_empty_spots()
        for next_move in next_empty:
            temp_game2 = TicTacToe(game.n)
            temp_game2.board = [row[:] for row in temp_game.board]
            temp_game2.current_player = player
            temp_game2.make_move(*next_move)
            if temp_game2.check_win() == player:
                return move
    return None

def get_llm_move(game, player, last_move, client, model):
    winning_move = find_winning_move(game, player)
    if winning_move:
        print(f"Player {player} found winning move: {winning_move}")
        return winning_move

    blocking_move = find_blocking_move(game, player)
    if blocking_move:
        print(f"Player {player} found blocking move: {blocking_move}")
        return blocking_move

    setup_move = find_setup_move(game, player)
    if setup_move:
        print(f"Player {player} found setup move: {setup_move}")
        return setup_move

    empty_spots = game.get_empty_spots()
    if not empty_spots:
        return None
    empty_spots_str = ", ".join([f"({r},{c})" for r, c in empty_spots])
    player_symbol = 'X' if player == 1 else 'O'
    opponent_symbol = 'O' if player == 1 else 'X'
    prompt = (
        f"You are Player {player} ({player_symbol}) in a {game.n}x{game.n} tic-tac-toe game. "
        f"Goal: Get {game.n} {player_symbol}'s in a row, column, or diagonal to win as soon as possible. "
        f"Current board (X=Player 1, O=Player 2, .=empty):\n{game.board_to_string()}\n"
        f"Opponent (Player {3-player}, {opponent_symbol}) last moved at {last_move if last_move else 'None'}. "
        f"Available empty spots: {empty_spots_str}. "
        f"Choose your next move to either WIN immediately, BLOCK Player {3-player} from winning, "
        f"or SET UP your next move to win. Prioritize winning over blocking over setup. "
        f"Return EXACTLY (row,col) in parentheses, e.g., (0,1), with no extra text."
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user","content": prompt}],
            max_tokens=10,
            temperature=0.5
        )
        move_str = response.choices[0].message.content.strip()
        move = parse_move(move_str)
        if move and move in empty_spots:
            print(f"Player {player} (Groq) Prompt:\n{prompt}\nMove: {move_str}")
            time.sleep(10)  # 6 RPM for Groq
            return move
        else:
            print(f"Groq invalid response: {move_str}. Using random move.")
    except Exception as e:
        print(f"Groq API error: {e}. Using random move.")
    return random_move(game)

def random_move(game):
    empty = game.get_empty_spots()
    return random.choice(empty) if empty else None

def run_trials(n, num_trials, llm1_client, llm1_model, llm2_client, llm2_model):
    results = []
    max_moves = n * n  # Ensure trials end even if no win
    for trial in range(num_trials):
        first_player = 1 if trial % 2 == 0 else 2
        print(f"\nStarting trial {trial + 1}/{num_trials} (Player {first_player} starts)")
        game = TicTacToe(n)
        game.current_player = first_player
        last_move = None
        move_count = 0
        while game.check_win() == 0 and move_count < max_moves:
            move = (get_llm_move(game, 1, last_move, llm1_client, llm1_model) if game.current_player == 1 
                    else get_llm_move(game, 2, last_move, llm2_client, llm2_model))
            if move and game.make_move(*move):
                last_move = move
                move_count += 1
            else:
                print(f"Trial {trial + 1}: No valid move after {move_count} moves")
                break
        winner = game.check_win()
        if winner == 0:  
            results.append(-1)
        elif winner == 1:  # LLM1 wins
            results.append(1)
        elif winner == 2:  # LLM2 wins
            results.append(0) 
        else:  # winner == -1 (draw)
            results.append(-1)
        print(f"Trial {trial + 1} result: {results[-1]} (1=LLM1, 0=LLM2, -1=Draw)")
    with open("Exercise1.json", "w") as f:
        json.dump({"trials": results}, f)
    return results

def plot_results(results):
    outcomes = np.array(results)
    llm1_wins = np.sum(outcomes == 1)
    llm2_wins = np.sum(outcomes == 0)
    draws = np.sum(outcomes == -1)
    print(f"Debug - LLM1 wins (1): {llm1_wins}, LLM2 wins (0): {llm2_wins}, Draws (-1): {draws}")
    plt.bar(['LLM1 Wins (1)', 'LLM2 Wins (0)', 'Draws (-1)'], [llm1_wins, llm2_wins, draws], 
            color=['green', 'red', 'blue'])
    plt.xlabel("Outcome")
    plt.ylabel("Count")
    plt.title(f"Tic-Tac-Toe Results ({len(results)} Trials, LLM1 Loses on Draw)")
    net_score = llm1_wins - draws
    plt.text(1, max(llm1_wins, llm2_wins, draws, 1) * 0.9, f"LLM1 Net Score: {net_score}", 
             ha='center', fontsize=10)
    plt.savefig("Exercise1_regular.png")
    plt.close()

def plot_binomial_distribution(results):
    n_trials = len(results)
    successes = sum(1 for r in results if r == 1)  # LLM1 wins (success)
    p = successes / n_trials if n_trials > 0 else 0  #prob
    k_values = np.arange(0, n_trials + 1)
    binom_pmf = binom.pmf(k_values, n_trials, p)

    plt.bar(k_values, binom_pmf, color='blue', alpha=0.7, label=f'Binomial PMF (p={p:.3f})')
    plt.axvline(x=successes, color='red', linestyle='--', label=f'Observed LLM1 Wins = {successes}')
    plt.xlabel("Number of LLM1 Wins (k)")
    plt.ylabel("Probability")
    plt.title(f"Binomial Distribution of LLM1 Wins ({n_trials} Trials)")
    plt.legend()
    plt.savefig("Exercise1.png")
    plt.close()

def human_vs_llm(n, llm_client, llm_model):
    game = TicTacToe(n)
    last_move = None
    while game.check_win() == 0:
        print(f"\nCurrent board:\n{game.board_to_string()}")
        if game.current_player == 1:
            while True:
                try:
                    move = input("Enter move (row,col) like '1,1': ").strip().split(",")
                    row, col = int(move[0]), int(move[1])
                    if game.make_move(row, col):
                        last_move = (row, col)
                        break
                    else:
                        print("Invalid move!")
                except (ValueError, IndexError):
                    print("please Use format 'row,col'.")
        else:
            move = get_llm_move(game, 2, last_move, llm_client, llm_model)
            if move and game.make_move(*move):
                last_move = move
    print(f"\nFinal board:\n{game.board_to_string()}")
    winner = game.check_win()
    if winner == 1:
        print("Human (X) wins!")
    elif winner == 2:
        print("LLM (O) wins!")
    else:
        print("Draw!")

if __name__ == "__main__":
    try:
        n = int(input("Enter board size (NxN, e.g., 3): "))
        if n < 2:
            raise ValueError("Give board size >= 2")

        # LLM1 (Groq)
        llm1_key = os.getenv("GROQ_API_KEY")
        if not llm1_key:
            llm1_key = input("Enter Groq API key for Player 1 (required if GROQ_API_KEY not set): ").strip()
            if not llm1_key:
                raise ValueError("No Groq API key provided for Player 1 and GROQ_API_KEY not set in .env")
        print(f"Using Groq API key for Player 1: {llm1_key[:5]}...")
        llm1_model = input("Enter Groq model for Player 1 (e.g., 'llama3-8b-8192'): ").strip()
        llm1_client = Groq(api_key=llm1_key)
        try:
            llm1_client.chat.completions.create(
                model=llm1_model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=10
            )
            print(f"Player 1 model '{llm1_model}' is accessible.")
        except Exception as e:
            print(f"Warning: Could not verify Player 1 model '{llm1_model}': {e}")

        # LLM2 (Groq)
        llm2_key = os.getenv("GROQ_API_KEY1")
        if not llm2_key:
            llm2_key = input("Enter Groq API key for Player 2 (required if GROQ_API_KEY1 not set): ").strip()
            if not llm2_key:
                raise ValueError("No Groq API key provided for Player 2 and GROQ_API_KEY1 not set in .env")
        print(f"Using Groq API key for Player 2: {llm2_key[:5]}...")
        llm2_model = input("Enter Groq model for Player 2 (e.g., 'mixtral-8x7b-32768'): ").strip()
        llm2_client = Groq(api_key=llm2_key)
        try:
            llm2_client.chat.completions.create(
                model=llm2_model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=10
            )
            print(f"Player 2 model '{llm2_model}' is accessible.")
        except Exception as e:
            print(f"Warning: Could not verify Player 2 model '{llm2_model}': {e}")

        mode = input("Enter mode (1: LLM vs LLM, 2: Human vs LLM): ").strip()
        if mode == "1":
            num_trials = int(input("Enter number of trials: "))
            print(f"Running {num_trials} trials for {n}x{n} board with LLM1 (Groq) vs LLM2 (Groq)...")
            results = run_trials(n, num_trials, llm1_client, llm1_model, llm2_client, llm2_model)
            llm1_wins = sum(1 for r in results if r == 1)
            llm2_wins = sum(1 for r in results if r == 0)
            draws = sum(1 for r in results if r == -1)
            print(f"LLM1 wins: {llm1_wins}, LLM2 wins: {llm2_wins}, Draws: {draws}")
            plot_results(results)
            plot_binomial_distribution(results)
            print("Results saved in Exercise1.json, regular plot in Exercise1_regular.png, binomial plot in Exercise1.png")
        elif mode == "2":
            human_vs_llm(n, llm2_client, llm2_model)
        else:
            print("Invalid mode!,Use 1 or 2.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")