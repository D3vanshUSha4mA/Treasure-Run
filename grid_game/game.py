import pygame
import sys
import random
import heapq
from rl_agent import DQNAgent

# Initialize pygame
pygame.init()

# Constants
ROWS, COLS = 15, 20
WIDTH, HEIGHT = 800, 600
TILE_SIZE = WIDTH // COLS
HEIGHT = TILE_SIZE * ROWS  # recalculate height to fit new rows
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class Game:
    def __init__(self, screen):
        self.screen = screen
        self.grid = self.get_static_maze()                #get the grid
        self.wall_img = pygame.image.load("assets/wall.png")             #load images
        self.floor_img = pygame.image.load("assets/floor.png")
        self.wall_img = pygame.transform.scale(self.wall_img, (TILE_SIZE, TILE_SIZE))
        self.floor_img = pygame.transform.scale(self.floor_img, (TILE_SIZE, TILE_SIZE))
        self.treasure_img = pygame.image.load("assets/treasure.png")
        self.treasure_img = pygame.transform.scale(self.treasure_img, (TILE_SIZE, TILE_SIZE))
        self.treasure_pos = (13, 18)
        self.player_img = pygame.image.load("assets/player.png")
        self.player_img = pygame.transform.scale(self.player_img, (TILE_SIZE, TILE_SIZE))

        self.player_pos = [1, 1]         #player position on grid
        self.player_pixel_pos = [1 * TILE_SIZE, 1 * TILE_SIZE]  #actual pixel position(for animation) 
        self.moving = False                                       #whether moving or not
        self.move_dir = (0, 0)                                    
        self.move_speed = 3  # pixels per frame
        self.key_img = pygame.image.load("assets/key.png").convert()
        self.key_img.set_colorkey((255, 255, 255))  # White becomes transparent
        self.key_img = pygame.transform.scale(self.key_img, (TILE_SIZE, TILE_SIZE))
        self.key_positions = [(1, 10), (9, 5), (5, 18)]
        self.collected_keys = []                                 
        self.correct_key = random.choice(self.key_positions)
        self.has_key=False
        self.held_key_pos=None                                #last position of key
        self.message = ""
        self.message_timer = 0  # Time left to display message (in frames)
        self.font = pygame.font.SysFont(None, 36)  # Default font, size 36
        self.won=False
        self.health=3
        self.max_health=3
        self.energy = 100
        self.max_energy = 100
        self.sprinting = False
        self.enemies=[[1, 10],[9, 5],[5, 18],[13, 18]]
        self.enemy_img = pygame.image.load("assets/enemy.png")
        self.enemy_img = pygame.transform.scale(self.enemy_img, (TILE_SIZE, TILE_SIZE))
        self.enemy_speed = 3 
        self.damage_cooldown=0
        self.enemy_pixel_pos=[[x[1]*TILE_SIZE,x[0]*TILE_SIZE]for x in self.enemies]
        self.enemy_targets=list(self.enemies)
        self.enemy_moving=[False]*len(self.enemies)
        self.enemy_speeds=[2]*len(self.enemies)
        self.game_over=False 
        self.enemy_chasing=[False]*len(self.enemies)
        self.chase_radius=6
        self.enemy_agent=DQNAgent(input_size=6,output_size=5)
        self.enemy_agent.load("enemy_model.pth")





    def get_state(self):                                          #returns a numerical vector that represents the current state of the game
        px,py=self.player_pos
        has_key=int(self.has_key)                                 #if key is held
        if self.held_key_pos:
            held_key_index=self.key_positions.index(self.held_key_pos) #index of key held
        else:
            held_key_index=-1
        tx,ty=self.treasure_pos
        dist_to_treasure=abs(px-tx)+abs(py-ty)                  #manhattan distance to the treasure
        health=self.health
        energy=self.energy
        enemy_data=[]
        for ex,ey in self.enemies:                              #enemy positions extended to list
            enemy_data.extend([ex,ey])
        state=[px,py,has_key,held_key_index,dist_to_treasure,health,energy]+enemy_data
        return state
    
    def get_enemy_state(self,i):                                #returns a normalized state vector
        ex,ey=self.enemies[i]
        px,py=self.player_pos
        visible=int(self.is_player_visible((ex,ey)))
        dist=abs(ex-px)+abs(ey-py)
        return [ex/ROWS,ey/COLS,px/ROWS,py/COLS,visible,dist/(ROWS+COLS)]
    



    def get_static_maze(self):
        
        return [
           
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1],
            [1,0,1,1,1,0,1,0,1,1,1,1,1,0,1,0,1,1,0,1],
            [1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1],
            [1,0,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1],
            [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
            [1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,1],
            [1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1],
            [1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
            [1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    ]

        

    def update(self): 
        self.handle_input()                                     #take player input
        self.animate_player()                                   #complete player movement animation
        current_pos=tuple(self.player_pos)
        if current_pos in self.key_positions:                   #check whether player is at key
            if not self.has_key:                                #if no key held already
                self.collected_keys.append(current_pos)         #append in collected
                self.has_key=True
                self.held_key_pos=current_pos
                print(f"🗝️ Collected key at {self.player_pos}")
            
        if current_pos==self.treasure_pos and self.has_key:     #if reaches the treasure with a key
            if self.held_key_pos==self.correct_key:             #if correct key win
                self.message = "🎉 Correct key! You won the game."
                self.won=True
                self.message_timer = 180  # 3 seconds
                self.has_key=False
                self.held_key_pos=None
            else:                                              #if wrong key,try again
                self.message = "❌ Wrong key! Try another one."
                self.message_timer = 120  # Show for 2 seconds at 60 FPS

                self.has_key=False
                self.held_key_pos=None
        self.update_enemies(train=True)                       #updates enemy decisions

        self.animate_enemies()                                #animate enemies
        self.check_enemy_collision()                          #check collision with player
        if self.damage_cooldown>0:
            self.damage_cooldown-=1
        if self.health <= 0:
            self.message = "☠️ Game Over! You lost."
            self.message_timer = 180  # Show message for 3 seconds
            self.game_over = True

        self.draw_grid()                                        #draw the grid
        if self.message_timer > 0:
            self.message_timer -= 1
        
        if (self.won or self.game_over) and self.message_timer == 0:
            pygame.time.delay(1000)
            pygame.quit()
            sys.exit()

        self.draw_bars()
        if len(self.enemy_agent.memory) > 64:
            self.enemy_agent.replay(64)
            self.enemy_agent.save("enemy_model.pth")           #save model after each replay
            



    def draw_grid(self):
        for row in range(ROWS):
            for col in range(COLS):
                tile_rect=pygame.Rect(col*TILE_SIZE,row*TILE_SIZE,TILE_SIZE,TILE_SIZE)
                if self.grid[row][col]==1:               #walls
                    self.screen.blit(self.wall_img,tile_rect)
                else:
                    self.screen.blit(self.floor_img,tile_rect)             #floor
                if (row,col)==self.treasure_pos:
                    self.screen.blit(self.treasure_img,tile_rect)
                if (row,col) in self.key_positions and (row,col) not in self.collected_keys:  #key,should not be in collected..
                    self.screen.blit(self.key_img,tile_rect)

        
        player_rect=pygame.Rect(self.player_pixel_pos[0],self.player_pixel_pos[1],TILE_SIZE,TILE_SIZE)
        self.screen.blit(self.player_img,player_rect)
        if self.message_timer > 0:
            msg_surface = self.font.render(self.message, True, (255, 255, 255))
            msg_rect = msg_surface.get_rect(center=(WIDTH // 2, HEIGHT - 30))
            self.screen.blit(msg_surface, msg_rect)
        for i in range(len(self.enemy_pixel_pos)):
            x, y = self.enemy_pixel_pos[i]
            rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)
            self.screen.blit(self.enemy_img, rect)

        




    def handle_input(self):
        if self.moving:
            return            #let the last input be completed,so return if already moving

        keys=pygame.key.get_pressed()
        row,col=self.player_pos
        dx,dy=0,0               #default dirxn
        next_tile=None
        if keys[pygame.K_UP]:
            next_tile=row-1,col
            dx,dy=0,-1
        elif keys[pygame.K_DOWN]:
            next_tile=row+1,col
            dx,dy=0,1
        elif keys[pygame.K_LEFT]:
            next_tile=row,col-1
            dx,dy=-1,0
        elif keys[pygame.K_RIGHT]:
            next_tile=row,col+1
            dx,dy=1,0

        if next_tile:
            x,y=next_tile
            if self.grid[x][y]==0:
                if [x,y] in self.enemies:                      #if an enemy is there,lose 1 health
                    if self.damage_cooldown==0:
                        self.health-=1
                        if self.health<=0:
                            self.game_over=True
                        self.health=max(0,self.health)
                        self.message="hit by enemy"
                        self.message_timer=60
                        self.damage_cooldown=60
                    return 

                
                
                if keys[pygame.K_LSHIFT] and self.energy>0:                 #for sprinting
                    self.move_speed=4
                    self.sprinting=True
                    self.energy-=2
                else:
                    self.move_speed=3
                    self.sprinting=False
                    self.energy+=0.5
                    self.energy=min(self.energy,self.max_energy)
                self.moving=True
                self.move_dir=(dx,dy)
        
    def animate_player(self):
        if not self.moving:
            return

        dx,dy=self.move_dir
        target_x=self.player_pos[1]*TILE_SIZE+dx*TILE_SIZE
        target_y=self.player_pos[0]*TILE_SIZE+dy*TILE_SIZE

        # Move player pixel-wise
        if self.player_pixel_pos[0]!=target_x:
            self.player_pixel_pos[0]+=self.move_speed*dx
        if self.player_pixel_pos[1]!=target_y:
            self.player_pixel_pos[1]+=self.move_speed*dy

        # Snap to grid when close enough
        if (abs(self.player_pixel_pos[0]-target_x)<self.move_speed and
            abs(self.player_pixel_pos[1]-target_y)<self.move_speed):
            self.player_pos[0]+=dy
            self.player_pos[1]+=dx
            self.player_pixel_pos[0]=self.player_pos[1]*TILE_SIZE
            self.player_pixel_pos[1]=self.player_pos[0]*TILE_SIZE
            self.moving=False

    def draw_bars(self):
    # Health Bar (3 red squares)
        for i in range(self.health):
            pygame.draw.rect(self.screen, (255, 0, 0), (10 + i * 25, 10, 20, 20))

        # Energy Bar Background
        pygame.draw.rect(self.screen, (100, 100, 100), (10, 40, 120, 15))  # background

        # Energy Bar Foreground (cyan)
        energy_width = int((self.energy / self.max_energy) * 120)
        pygame.draw.rect(self.screen, (0, 255, 255), (10, 40, energy_width, 15))

    def check_enemy_collision(self):
        if self.damage_cooldown>0:
            return 
        player_rect=pygame.Rect(self.player_pixel_pos[0],self.player_pixel_pos[1],TILE_SIZE,TILE_SIZE)
        for enemy_pos in self.enemies:
            enemy_rect=pygame.Rect(enemy_pos[1]*TILE_SIZE,enemy_pos[0]*TILE_SIZE,TILE_SIZE,TILE_SIZE)
            if player_rect.colliderect(enemy_rect):
                self.health-=1
                self.health=max(0,self.health)
                self.message="Hit by enemy"
                self.message_timer=60
                self.damage_cooldown=60
                break

                
    def a_star(self,start,goal):
        open_set=[]
        heapq.heappush(open_set,(0,start))     #priority queue : (f_score,position)
        came_from={}                           #to reconstruct path
        g_score={start:0}                      #distance from start to start is zero
        f_score={start:self.heuristic(start,goal)}
        visited=set()
        while open_set:
            current=heapq.heappop(open_set)[1]        #node with lowest f_score
            if current==goal:
                return self.reconstruct_path(came_from,current)     #goal reached,reconstruct path 
            visited.add(current)
            for neighbour in self.get_neighbours(current):
                if neighbour in visited:                             #skip visited ones
                    continue
                tentative_g=g_score[current]+1                       #all edge cost are 1
                if neighbour not in g_score or tentative_g<g_score[neighbour]:
                    came_from[neighbour]=current
                    g_score[neighbour]=tentative_g
                    f_score[neighbour]=tentative_g+self.heuristic(neighbour,goal)
                    heapq.heappush(open_set,(f_score[neighbour],neighbour))

        return []
    def heuristic(self,a,b):
        return abs(a[0]-b[0])+abs(a[1]-b[1])
    def get_neighbours(self,pos):
        x,y=pos
        neighbours=[]
        directions=[(-1,0),(1,0),(0,1),(0,-1)]
        for dx,dy in directions:
            nx,ny=x+dx,y+dy
            if  0<=nx<ROWS and 0<=ny<COLS and self.grid[nx][ny]==0:
                neighbours.append((nx,ny)) 
        return neighbours
    def reconstruct_path(self,came_from,current):
        path=[current]
        while current in came_from:
            current=came_from[current]
            path.append(current)
        path.reverse()
        return path
    def patrol_enemy(self,i):
        state=self.get_enemy_state(i)
        action=self.enemy_agent.select_action(state)            #use DQN agent to choose an action
        directions=[(-1,0),(1,0),(0,1),(0,-1),(0,0)]            #5 possible actions
        dx,dy=directions[action]
        row,col=self.enemies[i]
        nx,ny=row+dx,col+dy                                     #next possible tile
        if 0<=nx<ROWS and 0<=ny<COLS and self.grid[nx][ny]==0:
            if [nx,ny] not in self.enemy_targets:
                self.enemy_targets[i]=[nx,ny]                   #check already if not a enemy target
                self.enemy_moving[i]=True

        reward=-1                                               #default penalty
        px,py=self.player_pos
        dist=abs(px-nx)+abs(py-ny)
        if self.is_player_visible((nx,ny)):                     #reward is player is visible
            reward+=5
        if [nx,ny]==self.player_pos:                            #if enemy catches player
            reward+=50
        next_state=self.get_enemy_state(i)                      #store the transition in DQN replay memory  
        done=False
        self.enemy_agent.remember(state,action,reward,next_state,done)

    def is_player_visible(self,enemy_pos):     #check is the player is visible.....
        ex,ey=enemy_pos
        px,py=self.player_pos
        if ex==px:
            for y in range(min(ey,py)+1,max(ey,py)):
                if self.grid[ex][y]==1:
                    return False
            return True
        if ey==py:
            for x in range(min(ex,px)+1,max(ex,px)):
                if self.grid[x][ey]==1:
                    return False
            return True
        return False
    def chase_player(self,enemy_index):
        start=tuple(self.enemies[enemy_index])       #current position of enemy as a tuple
        goal=tuple(self.player_pos)                  #current position of player as a tuple
        path=self.a_star(start,goal)                 #compute the shortest path
        if len(path)>1:                              #if valid path
            next_step=path[1]                        #take next step towards the player
            self.enemy_targets[enemy_index]=list(next_step)   #set position for animation
            self.enemy_moving[enemy_index]=True      #set moving as true  

    def update_enemies(self,train=False):
        for i,enemy in enumerate(self.enemies):
            if self.enemy_moving[i]:                #if enemy already moving to a tile....
                continue                             
            enemy_pos=self.enemies[i]               #get current grid postions of enemy and player
            player_pos=self.player_pos
            dist=abs(player_pos[0]-enemy_pos[0])+abs(player_pos[1]-enemy_pos[1])      #manhattan distance.....
            if self.is_player_visible(enemy_pos):                  #if player is visible
                self.enemy_chasing[i]=True                         #start chasing
                self.chase_player(i)                               #updates target using A*
            elif self.enemy_chasing[i]:        
                if dist<=self.chase_radius:                        #if still inside chase radius continue chasing
                    self.chase_player(i)
                else:
                    self.enemy_chasing[i]=False                    #stop chasing and patrol
                    self.patrol_enemy(i)
            else:
                self.patrol_enemy(i)
    def animate_enemies(self):
        for i in range(len(self.enemies)):
            if not self.enemy_moving[i]:               #if enemy not moving currently,move to next enemy..
                continue

            target_x=self.enemy_targets[i][1]*TILE_SIZE              #converts enemy target tile to pixel coordinates
            target_y=self.enemy_targets[i][0]*TILE_SIZE
            px,py=self.enemy_pixel_pos[i]                            #enemy current pixel position
            speed=self.enemy_speeds[i]                               #enemy speed in pixels per frame

            if px<target_x:                             #movement toward x target
                px+=speed
            elif px>target_x:
                px-=speed

            if py<target_y:                            #movement toward y target
                py+=speed
            elif py>target_y:
                py-=speed

            self.enemy_pixel_pos[i]=[px,py]           #update enemy's new pixel position after moving

            if abs(px-target_x)<=speed and abs(py-target_y)<=speed:     #snapping condition
                self.enemy_pixel_pos[i]=[target_x,target_y]
                self.enemies[i]=list(self.enemy_targets[i])  # Update logical position only now
                self.enemy_moving[i]=False

    def step(self, action):
        move_map = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        if action < len(move_map):
            dx, dy = move_map[action]
            new_x = self.player_pos[0] + dx
            new_y = self.player_pos[1] + dy
            if 0 <= new_x < ROWS and 0 <= new_y < COLS and self.grid[new_x][new_y] == 0:
                self.player_pos = [new_x, new_y]
                self.player_pixel_pos = [new_y * TILE_SIZE, new_x * TILE_SIZE]

        reward = 0
        if tuple(self.player_pos) in self.key_positions and not self.has_key:
            self.collected_keys.append(tuple(self.player_pos))
            self.has_key = True
            self.held_key_pos = tuple(self.player_pos)
            reward += 10

        if tuple(self.player_pos) == self.treasure_pos:
            if self.has_key and self.held_key_pos == self.correct_key:
                reward += 100
                self.won = True
            elif self.has_key:
                reward -= 10
                self.has_key = False
                self.held_key_pos = None

        return self.get_state(), reward, self.won
        
    def reset(self):
        # Reset player
        self.player_pos = [1, 1]
        self.player_pixel_pos = [1 * TILE_SIZE, 1 * TILE_SIZE]
        self.has_key = False
        self.held_key_pos = None
        self.collected_keys = []
        self.correct_key = random.choice(self.key_positions)

        # Reset game state
        self.health = self.max_health
        self.energy = self.max_energy
        self.won = False
        self.game_over = False
        self.damage_cooldown = 0
        self.message = ""
        self.message_timer = 0

        # Reset enemies
        self.enemies = [[1, 10], [9, 5], [5, 18], [13, 18]]
        self.enemy_pixel_pos = [[x[1] * TILE_SIZE, x[0] * TILE_SIZE] for x in self.enemies]
        self.enemy_targets = list(self.enemies)
        self.enemy_moving = [False] * len(self.enemies)
        self.enemy_chasing = [False] * len(self.enemies)

        return self.get_state()

                

