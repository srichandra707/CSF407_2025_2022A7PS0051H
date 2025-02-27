import numpy as np
from random import randint
class wumpusworld:
    def __init__(self,n):
        self.n=n
        self.pit_probability=0.2
        self.pits=int((n**2-1)*self.pit_probability) #why n^2-1 ? because the starting square cannot have a pit
        self.wumpus=1
        self.gold=1
        self.start=[1,1] #this is in cartesian coordinates, otherwise it'd have been [n-1,0]
        self.world=self.create_world(n)

    def create_world(self, n):
        temp_world=np.zeros((n, n), dtype=object)
        temp_world[self.n-1][0]='A'
        components=[]
        while len(components)<self.pits:
            row=randint(0,n-1)
            col=randint(0,n-1)
            if row!=0 and col!=0 and temp_world[row][col]==0:
                temp_world[row][col]='P'
                components.append(['P',[row,col]])
            print(temp_world)
            print()
            print(components)
            print()
        while len(components)<self.pits+self.wumpus:
            row=randint(0,n-1)
            col=randint(0,n-1)
            if row!=0 and col!=0 and temp_world[row][col]==0:
                temp_world[row][col]='W'
                components.append(['W',[row,col]])
            print(temp_world)
            print()
            print(components)
            print()
        while len(components)<self.pits+self.wumpus+self.gold:
            row=randint(0,n-1)
            col=randint(0,n-1)
            if row!=0 and col!=0 and temp_world[row][col]==0:
                temp_world[row][col]='G'
                components.append(['G',[row,col]])
            print(temp_world)
            print()
            print(components)
            print()
        for t,pos in components:
            if t=='G':
                continue
            if pos[0]+1<n:
                self.create_stench_and_breeze(temp_world,pos[0]+1,pos[1],t=='W')
            if pos[0]-1>=0:
                self.create_stench_and_breeze(temp_world,pos[0]-1,pos[1],t=='W')
            if pos[1]+1<n:
                self.create_stench_and_breeze(temp_world,pos[0],pos[1]+1,t=='W')
            if pos[1]-1>=0:
                self.create_stench_and_breeze(temp_world,pos[0],pos[1]-1,t=='W')
            

        return temp_world
    
    def create_stench_and_breeze(self,world,row,col,stench):
        if stench:
            if world[row][col]==0:
                world[row][col]='S'
            else:
                if 'S' not in world[row][col] and world[row][col]!='P':
                    world[row][col]+=',S'
            print(world)
            print()
        else:
            if world[row][col]==0:
                world[row][col]='B'
            else:
                if 'B' not in world[row][col] and world[row][col]!='P' and world[row][col]!='W':
                    world[row][col]+=',B'
            print(world)
            print()
    def get_cartesian_coordinates(self,world,row,col):
        return world[self.n-row][col-1]

    def cell_contains(self,world,row,col):
        return self.get_cartesian_coordinates(world,row,col).split(',')
    def print_world(self):
        print("\n" + "-" * (self.n * 4 + 1))
        for row in self.world:
            formatted_row = '|'.join([f"{cell:<3}" if cell else '   ' for cell in row])
            print(f"|{formatted_row}|")
            print("-" * (self.n * 4 + 1))
        
if __name__=="__main__":
    print("Enter the size of the world: ")
    n=int(input())
    world=wumpusworld(n)
    world.print_world()