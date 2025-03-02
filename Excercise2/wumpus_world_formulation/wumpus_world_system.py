import numpy as np
from random import randint
import matplotlib.pyplot as plt

class wumpusworld:
    def __init__(self, n):
        self.n = n
        self.pit_probability = 0.2
        self.pits = int((n ** 2 - 1) * self.pit_probability)
        self.wumpus_probability = 1 / (n ** 2 - 1)
        self.start = [n - 1, 0]
        self.world = self.create_world(n)

    def create_world(self, n):
        world = np.zeros((n, n), dtype=object)
        world[self.start[0]][self.start[1]] = 'A'
        components = []
        def place_component(symbol, count):
            while len(components) < count:
                row, col = randint(0, n - 1), randint(0, n - 1)
                if (row, col) in [(n-1, 0), (n-1, 1), (n-2, 0)]:
                    continue
                if world[row][col] == 0:
                    world[row][col] = symbol
                    components.append((symbol, [row, col]))

        place_component('P', self.pits)
        place_component('W', self.pits+1)
        print(f"placed wumpus at {components[-1][1]}")
        def is_safe_path(start,end,world):
            directions=[(0,1),(0,-1),(1,0),(-1,0)]
            visited=set()
            def dfs(pos):
                if pos==end:
                    return True
                visited.add(pos)
                for dr,dc in directions:
                    r,c=pos[0]+dr,pos[1]+dc
                    if 0<=r<n and 0<=c<n and world[r][c] not in ['P','W'] and (r,c) not in visited:
                        if dfs((r,c)):
                            return True
                return False
            return dfs(start)
        def place_gold():
            while True:
                row,col=randint(0,n-1),randint(0,n-1)
                if (row,col)!=self.start and world[row][col]==0:
                    if is_safe_path(tuple(self.start),(row,col),world):
                        world[row][col]='G'
                        components.append(('G',[row,col]))
                        print(f"Gold placed at {row},{col}")
                        break
        place_gold()

        for comp, pos in components:
            if comp == 'G':
                continue
            self.create_stench_and_breeze(world, pos, comp == 'W')
        return world
    def get_cartesian_coordinates(self, world, row, col):
        return world[self.n - row][col - 1]

    def cell_contains(self, world, row, col):
        return self.get_cartesian_coordinates(world, row, col).split(',')

    def create_stench_and_breeze(self, world, pos, is_wumpus):
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in directions:
            r, c = pos[0] + dr, pos[1] + dc
            if 0 <= r < self.n and 0 <= c < self.n:
                if is_wumpus:
                    if world[r][c]==0:
                        world[r][c]='S'
                    else:
                        if 'S' not in world[r][c] and world[r][c]!='P':
                            world[r][c]+=',S'
                else:
                    if world[r][c]==0:
                        world[r][c]='B'
                    else:
                        if 'B' not in world[r][c] and world[r][c]!='P' and world[r][c]!='W':
                            world[r][c]+=',B'

    def print_world(self):
        print("\n" + "-" * (self.n * 4 + 1))
        for row in self.world:
            formatted_row = '|'.join([f"{cell:<3}" if cell else '   ' for cell in row])
            print(f"|{formatted_row}|")
            print("-" * (self.n * 4 + 1))

    def save_world_as_png(self, filename="world.png"):
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-0.5, self.n - 0.5)
        ax.set_ylim(-0.5, self.n - 0.5)
        
        for i in range(self.n):
            ax.axhline(i - 0.5, color='black', linewidth=0.5)
            ax.axvline(i - 0.5, color='black', linewidth=0.5)
        
        for i in range(self.n):
            for j in range(self.n):
                cell_content = self.world[self.n - i - 1][j]  
                if cell_content:
                    ax.text(j, i, str(cell_content), ha='center', va='center', fontsize=10)

        ax.set_xticks(range(self.n))
        ax.set_yticks(range(self.n))
        ax.set_xticklabels(range(1, self.n + 1))
        ax.set_yticklabels(range(1, self.n + 1))
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        plt.savefig(filename)
        plt.close()

    def save_world_as_txt(self, filename="world.txt"):
        with open(filename, 'w') as f:
            for row in self.world:
                line = ' '.join(str(cell) if cell else '0' for cell in row)  
                f.write(line + '\n')

if __name__ == "__main__":
    print("Enter the size of the world: ")
    n = int(input())
    world = wumpusworld(n)
    world.print_world()