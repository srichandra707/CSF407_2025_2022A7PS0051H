import sys
import pandas as pd
sys.setrecursionlimit(10000)
import numpy as np
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import random
from random import choice
import itertools
from wumpus_world import wumpusworld  
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

def create_output_directory(base_dir="output_excercise2"):
    timestamp=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir=os.path.join(base_dir,f"run_{timestamp}")
    os.makedirs(output_dir,exist_ok=True)
    return output_dir

class WumpusBN:
    def __init__(self,n):
        self.n=n
        self.model=BayesianNetwork()
        self.inference=None
        self.evidence={}
        self.create_network_structure()
        self.add_probability_tables()
        self.risk_map=np.zeros((self.n, self.n))

    def create_network_structure(self):
        edges=[]
        
        for i in range(self.n):
            for j in range(self.n):
                
                for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    xi,xj=i+dx,j+dy
                    if 0<=xi<self.n and 0<=xj<self.n:
                        edges.append((f"P_{i}_{j}",f"B_{xi}_{xj}"))
                
                for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    xi,xj=i+dx,j+dy
                    if 0<=xi<self.n and 0<=xj<self.n:
                        edges.append((f"W_{i}_{j}",f"S_{xi}_{xj}"))
        self.model.add_edges_from(edges)

    def add_probability_tables(self):
        cpds=[]
        for i in range(self.n):
            for j in range(self.n):
                if (i,j)==(0,0):
                    cpds.append(TabularCPD(f"P_{i}_{j}",2,[[1.0],[0.0]]))
                else:
                    cpds.append(TabularCPD(f"P_{i}_{j}",2,[[0.8],[0.2]]))

        wumpus_prior=1/(self.n**2-1)
        for i in range(self.n):
            for j in range(self.n):
                if (i,j)==(0,0):
                    cpds.append(TabularCPD(f"W_{i}_{j}",2,[[1.0],[0.0]]))
                else:
                    cpds.append(TabularCPD(f"W_{i}_{j}", 2, [[1-wumpus_prior],[wumpus_prior]]))

        
        for i in range(self.n):
            for j in range(self.n):
                
                pit_parents=self.get_adjacent_nodes(i, j, "P")
                cpds.append(self.create_or_cpd(f"B_{i}_{j}",pit_parents))

                
                wumpus_parents=self.get_adjacent_nodes(i, j,"W")
                cpds.append(self.create_or_cpd(f"S_{i}_{j}",wumpus_parents))

        self.model.add_cpds(*cpds)
        self.inference=VariableElimination(self.model)

    def create_or_cpd(self,node,parents):        
        if not parents:
            return TabularCPD(node,2,[[1.0],[0.0]])

        values=[[],[]]
        for combo in itertools.product([0,1],repeat=len(parents)):
            values[0].append(1 if sum(combo)==0 else 0)
            values[1].append(0 if sum(combo)==0 else 1)

        return TabularCPD(node, 2, values,evidence=parents,evidence_card=[2]*len(parents))

    def get_adjacent_nodes(self,i,j,prefix):
        parents=[]
        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            xi,xj=i+dx,j+dy
            if 0<=xi<self.n and 0<=xj<self.n:
                parents.append(f"{prefix}_{xi}_{xj}")
        return parents

    def update_evidence(self,cell,has_breeze,has_stench):
        self.evidence[f"B_{cell[0]}_{cell[1]}"]=int(has_breeze)
        self.evidence[f"S_{cell[0]}_{cell[1]}"]=int(has_stench)

    def get_pit_risk(self,current_pos,visited):
        max_depth=self.n//2 -1
        epsilon=1e-10

        for i in range(self.n):
            for j in range(self.n):
                if (i,j) not in visited and self.is_reachable(current_pos, (i, j), max_depth):
                    # pit_prob = self.inference.query(
                    #     [f"P_{i}_{j}"], evidence=self.evidence
                    # ).values[1]
                    # print(self.inference.query(
                    #     [f"P_{i}_{j}"], evidence=self.evidence
                    # ))
                    # wumpus_prob = self.inference.query(
                    #     [f"W_{i}_{j}"], evidence=self.evidence
                    # ).values[1]
                    # print(self.inference.query(
                    #     [f"W_{i}_{j}"], evidence=self.evidence
                    # ))
                    # print((i, j))
                    # print(pit_prob + wumpus_prob - (pit_prob * wumpus_prob))
                    # self.risk_map[i, j] = pit_prob + wumpus_prob - (pit_prob * wumpus_prob)
                    
                    pit_dist=self.inference.query([f"P_{i}_{j}"],evidence=self.evidence)
                    pit_values=pit_dist.values.copy()
                    if not np.isnan(pit_values).any():
                        pit_sum=pit_values.sum()
                        pit_prob=pit_values[1]/(pit_sum+epsilon)
                    else:
                        pit_prob=0.7  # Mark as high risk if probability is undefied

                    self.risk_map[i,j]=pit_prob

        return self.risk_map

    def get_combined_risk(self, current_pos, visited):        
        max_depth = self.n // 2 - 1
        epsilon = 1e-10

        # print(self.evidence)

        for i in range(self.n):
            for j in range(self.n):
                if (i,j) not in visited and self.is_reachable(current_pos, (i, j), max_depth):
                    # pit_prob = self.inference.query(
                    #     [f"P_{i}_{j}"], evidence=self.evidence
                    # ).values[1]
                    # print(self.inference.query(
                    #     [f"P_{i}_{j}"], evidence=self.evidence
                    # ))
                    # wumpus_prob = self.inference.query(
                    #     [f"W_{i}_{j}"], evidence=self.evidence
                    # ).values[1]
                    # print(self.inference.query(
                    #     [f"W_{i}_{j}"], evidence=self.evidence
                    # ))
                    # print((i, j))
                    # print(pit_prob + wumpus_prob - (pit_prob * wumpus_prob))
                    # self.risk_map[i, j] = pit_prob + wumpus_prob - (pit_prob * wumpus_prob)

                    pit_dist = self.inference.query([f"P_{i}_{j}"], evidence=self.evidence)
                    pit_values = pit_dist.values.copy()
                    if not np.isnan(pit_values).any():
                        pit_sum = pit_values.sum()
                        pit_prob = pit_values[1] / (pit_sum + epsilon)
                    else:
                        pit_prob = 0.8  

                    
                    wumpus_dist = self.inference.query([f"W_{i}_{j}"], evidence=self.evidence)
                    wumpus_values = wumpus_dist.values.copy()
                    if not np.isnan(wumpus_values).any():
                        wumpus_sum = wumpus_values.sum()
                        wumpus_prob = wumpus_values[1] / (wumpus_sum + epsilon)
                    else:
                        wumpus_prob = 0.8

                    self.risk_map[i, j] = pit_prob + wumpus_prob - (pit_prob * wumpus_prob)

        return self.risk_map

    def is_reachable(self, start, target, max_depth):
        
        return (abs(start[0] - target[0]) + abs(start[1] - target[1])) <= max_depth

class WumpusAgent:
    def __init__(self, n, bn_model):
        self.n = n
        self.bn = bn_model
        self.current_pos = (n - 1, 0)  
        self.visited = {self.current_pos}
        self.safe_positions = [self.current_pos]

    def get_valid_moves(self):
        
        x, y = self.current_pos
        return [(x + dx, y + dy) for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]
                if 0 <= x + dx < self.n and 0 <= y + dy < self.n]

    def make_move(self, strategy='bayesian'):

        
        valid_moves = self.get_valid_moves()

        
        unvisited_moves = [move for move in valid_moves if move not in self.visited]

        if strategy == 'random':
            return choice(valid_moves)

        risk_map = self.bn.get_combined_risk(self.current_pos, self.visited)
        if unvisited_moves:
            if min(unvisited_moves, key=lambda pos: risk_map[pos]) != 1:
                
                min_risk = min(risk_map[pos] for pos in unvisited_moves)
                
                min_risk_moves = [pos for pos in unvisited_moves if risk_map[pos] == min_risk]
                
                return choice(min_risk_moves)
            else:
                
                min_risk = min(risk_map[pos] for pos in valid_moves)
                min_risk_moves = [pos for pos in valid_moves if risk_map[pos] == min_risk]
                return choice(min_risk_moves)

        min_risk = min(risk_map[pos] for pos in valid_moves)
        min_risk_moves = [pos for pos in valid_moves if risk_map[pos] == min_risk]
        return choice(min_risk_moves)

    def mark_safe(self):
        
        x, y = self.current_pos
        self.bn.evidence[f"P_{x}_{y}"] = 0  
        self.bn.evidence[f"W_{x}_{y}"] = 0  
        self.bn.risk_map[x, y] = 0

    def handle_death(self, cell_str):
        
        x, y = self.current_pos
        if "P" in cell_str:
            self.bn.evidence[f"P_{x}_{y}"] = 1  
        if "W" in cell_str:
            self.bn.evidence[f"W_{x}_{y}"] = 1  
        self.bn.risk_map[x, y] = 1
        if len(self.safe_positions) > 1:
            self.current_pos = self.safe_positions[-2]
            self.safe_positions.pop()

class EnhancedWumpusWorld(wumpusworld):
    def __init__(self, n):
        super().__init__(n)
        n = n if n > 0 else 10
        wumpusworld.print_world(self)
        self.bn = WumpusBN(n)
        self.agent = WumpusAgent(n, self.bn)
        self.step = 0
        self.output_directory = create_output_directory()
        wumpusworld.save_world_as_png(self, f"{self.output_directory}/world.png")
        wumpusworld.save_world_as_txt(self, f"{self.output_directory}/world.txt")

    def run_simulation(self):
        
        while True:
            x, y = self.agent.current_pos
            cart_x = self.n - x
            cart_y = y + 1
            cell_content = self.get_cartesian_coordinates(self.world, cart_x, cart_y)
            cell_str = str(cell_content)
            
            self.bn.update_evidence((x, y),'B' in cell_str,'S' in cell_str)

            
            self.agent.mark_safe()

            
            self.visualize_risk()

            if 'G' in cell_str:
                print(f"Gold found at step {self.step}!")
                break
            if 'P' in cell_str or 'W' in cell_str:
                print(cell_str)
                print(f"Agent died at step: {self.step}! Restarting...")
                self.agent.handle_death(cell_str)
                continue
            
            strategy = 'bayesian'
            next_pos = self.agent.make_move(strategy)

            if next_pos:
                self.agent.current_pos = next_pos
                self.agent.visited.add(next_pos)
                self.agent.safe_positions.append(next_pos)
                self.step += 1

    def visualize_risk(self):
        risk_map = self.bn.get_combined_risk(self.agent.current_pos, self.agent.visited)

        plt.figure(figsize=(8, 6))
        plt.imshow(risk_map, cmap='Reds', vmin=0, vmax=1)
        plt.colorbar(label='Combined Risk (Pit + Wumpus)')
        plt.scatter(self.agent.current_pos[1], self.agent.current_pos[0],
                    c='blue', s=200, edgecolors='white')
        plt.title(f"Step {self.step} - Current Position {self.agent.current_pos}")
        file_path = os.path.join(self.output_directory, f"risk_map_step_{self.step}.png")
        plt.savefig(file_path)
        plt.close()

if __name__ == "__main__":
    n = int(input("Enter grid size (>=4): "))
    world = EnhancedWumpusWorld(n)
    world.run_simulation()
