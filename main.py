
import pygame, sys, time
import numpy as np
from pygame.locals import *
import random as r
from random import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

pygame.init()
clock = pygame.time.Clock()

torch.set_default_dtype(torch.float64)

i = np.array([1,0,0])
j = np.array([0,1,0])
k = np.array([0,0,1])
origin = np.array([0,0,0])
gravity = np.array([0,0,-9.8])
dt = 0.03

def g(x, y):
    return torch.where(y >= 0, (1+x) * y, (1-x) * y)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=0.01)
        m.bias.data.fill_(0.01)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def flatten2d(arr):
    out = []
    for x in arr:
        for y in x:
            out.append(y)
    return out

class network(nn.Module):
    def __init__(self,layers,lrate,is_distribution = False):
      super(network, self).__init__()
      self.distribution = is_distribution
      self.network = self.constructnet(layers)
      self.optimizer = optim.Adam(self.parameters(),lr = lrate)

    
    def constructnet (self,layers):
      arcitecture = []
      for x in range(len(layers) - 1):
        arcitecture.append(nn.Linear(layers[x],layers[x+1]))
        if x != len(layers) - 2:
         arcitecture.append(nn.ReLU())
      if self.distribution:
        arcitecture.append(nn.Softmax(dim = -1))
      return nn.Sequential(*arcitecture)

    def forward (self,x):
        if self.distribution:
            distribution = self.network(x)
            distribution = Categorical(distribution)
            return distribution
        return self.network(x)
     
    def saveparameters (self,filepath):
        torch.save(self.state_dict(),filepath)
    
    def loadparameters (self,filepath):
        self.load_state_dict(torch.load(filepath))


class memory:
    def __init__ (self):
        self.length = 0
        self.states = []
        self.actions = []
        self.probabilities = []
        self.values = []
        self.rewards = []
        self.trajectories = []
    
    def generatetrajectories(self):
        return self.trajectories
        
    def store (self,state,action,probability,value,reward):
        self.length += 1
        self.states.append(state)
        self.actions.append(action)
        self.probabilities.append(probability)
        self.values.append(value)
        self.rewards.append(reward)
    
    def storetrajectory(self):
        self.trajectories.append([
             np.array(self.states),
             np.array(self.actions),
             np.array(self.probabilities),
             np.array(self.values),
             np.array(self.rewards)
             ])
        self.clear()
    
    def clear(self,clear_trajectories = False):
        self.length = 0
        self.states = []
        self.actions = []
        self.probabilities = []
        self.values = []
        self.rewards = []
        if clear_trajectories:
            self.trajectories = []


class agent:
    def __init__ (self):
        self.policynet = network([13,256,256,5],0.0001,is_distribution = True)
        self.valuenet = network([13,256,256,1],0.0001,)
        self.policynet.apply(init_weights)
        self.memory = memory()
        self.record = float("-inf")

    def storememory(self,state,action,probability,value,reward):
        self.memory.store(state,action,probability,value,reward)
    
    def endtrajectory(self):
        self.memory.storetrajectory()

    def saveparameters (self,policyfile,valuefile):
        self.policynet.saveparameters(policyfile)
        self.valuenet.saveparameters(valuefile)
    
    def loadparameters(self,policyfile,valuefile):
        self.policynet.loadparameters(policyfile)
        self.valuenet.loadparameters(valuefile)

    def getaction(self,inputstate,show_probs = False):
        state = torch.tensor(np.array([inputstate]),dtype =torch.float64)
        distribution = self.policynet(state)
        if(show_probs):
            return distribution.probs
        value = self.valuenet(state)
        action = distribution.sample()

        return torch.squeeze(action).item(),\
               torch.squeeze(distribution.log_prob(action)).item(),\
               torch.squeeze(value).item()

    def train(self,iterations,epsilon = 0.1,discount = 0.99):
        data = self.memory.generatetrajectories()
        states_ = [x[0] for x in data]
        actions_ = [x[1] for x in data]
        probs_ = [x[2] for x in data]
        values_  = [x[3] for x in data]
        rewards__ = [x[4] for x in data]
        totalrewards = np.sum(flatten2d(rewards__))
        if totalrewards > self.record:
            self.record = totalrewards
            print("record rewards acheived")
            #self.saveparameters("recordpolicy.pth","recordvalue.pth")
        futurerewards = []
        advantages = []
        

        for i in range(len(rewards__)):

            rewards_ = rewards__[i]
            vals_ = values_[i]
            
            # calculate reward esitmates
            rewards_to_go_ = np.zeros(len(rewards_))  

            for T in range (len(rewards_to_go_)):
                discount_ = 1
                R = 0
                for t in range (T,len(rewards_to_go_)):
                    R += discount_ * rewards_[t]
                    discount_ = discount_ * discount
                rewards_to_go_[T] = R

            futurerewards.append(rewards_to_go_)

            # calculate advantage estimates
            advantages_ = np.zeros(len(rewards_))

            for T in range (len(advantages_) - 1):
                discount_ = 1
                A = 0
                for t in range(T,len(advantages_ ) - 1):
                    A+= discount_ * (rewards_[t] + discount * vals_[t+1] - vals_[t])
                    discount_ = discount_ * discount
                advantages_[T] = A
                
            advantages.append(advantages_)
            

        
        states = torch.tensor(flatten2d(states_))
        oldprobs = torch.tensor(flatten2d(probs_))
        actions = torch.tensor(flatten2d(actions_))
        rewards_to_go = torch.tensor(flatten2d(futurerewards))
        advantages = torch.tensor(flatten2d(advantages))


        # training on policy
        
        for _ in range (iterations):
            self.policynet.optimizer.zero_grad()
            distribution = self.policynet(states) 
            newprobs = distribution.log_prob(actions)
            ratio = (newprobs - oldprobs).exp()
            PPOLoss = -1 * torch.min(advantages * ratio, g(epsilon,advantages)).mean()
            PPOLoss.backward()
            self.policynet.optimizer.step()

        # trainig on value estimator
        for _ in range (iterations):
            self.valuenet.optimizer.zero_grad()
            networkvalues = self.valuenet(states)
            valueloss = ((networkvalues - rewards_to_go) **2).mean()
            weightedvalueloss = valueloss * 0.5
            weightedvalueloss.backward()
            self.valuenet.optimizer.step()
        
        self.memory.clear(clear_trajectories = True)



def draw_text(surface, text, color, position,size):
    if not pygame.font.get_init():
        pygame.font.init()
    font = pygame.font.Font(None, size)
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, position)

xdim = 1500
ydim = 1000
screen = pygame.display.set_mode((xdim, ydim))

def transform(p):
    x = p[0]
    y = p[1]

    return (xdim/2 + x,ydim/2 + y)

def length(v):
    return np.linalg.norm(v)

def normalize(v):
    if length(v) == 0:
        return np.array([0,0,0])
    return v / length(v)

def project(v,w):
    wnorm = normalize(w)
    proj = np.dot(v,wnorm)
    return wnorm * proj

def cross(v,w):
    return np.cross(v,w)

def project_to_screen(n,w,v,p):
    vector = v - p
    norm_direction = normalize(n)
    if np.dot(vector,norm_direction) < 1:
        return np.array([0,0])
    if np.all(v == p):
        return(np.array([0,0]))
    v1 = project(vector,n)
    if length(v1) == 0:
        return(np.array([0,0]))
    q = normalize(cross(w,n))
    scalefactor = length(n) / length(v1)
    vector = vector * scalefactor
    vector = vector - n
    xout = np.dot(project(vector,w),w) 
    yout = np.dot(project(vector,q),q)
    return np.array([xout,yout])

def rotate(vector, axis, angle):
    axis = normalize(axis)  
    if abs(np.dot(normalize(vector), axis) - 1) < 1e-6:
        return vector
    projection = project(vector, axis)
    normal = vector - projection
    mag = length(normal)
    if mag > 1e-6:  
        normal_unit = normal / mag
        cross_unit = cross(axis, normal_unit)  
        rotated_normal = mag * (normal_unit * np.cos(angle) + cross_unit * np.sin(angle))
        return projection + rotated_normal
    else:
        return vector

def cos(x):
    return np.cos(x)


def sin(x):
    return np.sin(x)


def spherepoint(a,b,r,x,y,z):
    return (x + r * cos(b) * cos(a), y + r * cos(b) * sin(a), r * sin(b))


class polygon:

    def __init__(self, vertices, color):
        self.vertices = vertices # array of length 3 tuple
        self.color = color # (r,g,b)

    def setcolor(self,color):
        self.color = color

    

class environment:

    def __init__(self,background_color = (0,0,0)):
        self.backdrop = background_color
        self.polygons = []

    def add(self,polygon):
        self.polygons.append(polygon)
    
    def getobjects(self):
        return self.polygons

    def clear(self):
        self.polygons = []


 
class viewer:

    def __init__(self,initial_position = (0,0,0), view_direction = (1000,0,0), tilt_direction = (0,1,0)):
        self.position = np.array(initial_position)
        self.direction = np.array(view_direction)
        self.tilt = np.array(tilt_direction)

    def renderpoint(self,point):
        return project_to_screen(self.direction,self.tilt,point,self.position)

    def renderpolygon(self,polygon):
        out = []
        for p in polygon.vertices:
            point = self.renderpoint(np.array(p))
            if point[0] == 0 and point[1] == 0:
                return []
            point = transform(tuple(point))
            out.append(point)
        return out

    def renderworld(self,world):
        screen.fill(world.backdrop)
        for polygon in world.getobjects():
           vertices = self.renderpolygon(polygon)
           if len(vertices) > 0:
            pygame.draw.polygon(screen, polygon.color,vertices)
        pygame.draw.line(screen,(255,0,0),(xdim/2 - 20, ydim/2),(xdim/2 + 20,ydim/2))
        pygame.draw.line(screen,(255,0,0),(xdim/2 , ydim/2 - 20),(xdim/2 ,ydim/2 + 20))

    def translate(self,v):
        self.position = self.position + v
    
    def rotate_horizontal(self,a):
        proj = project(self.direction,k) 
        w = self.direction - proj
        mag = length(w)
        normal = -1 * normalize(cross(k,w))
        new = normalize(w) * np.cos(a) + normal * np.sin(a)
        self.direction= mag * new + proj
        self.tilt = -1  * normalize(cross(self.direction,k))

       
    
    def rotate_vertical(self,a):
        temp = normalize(cross(self.tilt,self.direction))
        mag = length(self.direction)
        newdirection = normalize(self.direction) * np.cos(a) + temp * np.sin(a)
        self.direction = mag * newdirection

    def zoom_in(self):
        new = self.direction + 10 *normalize(self.direction)
        self.direction = new


    def zoom_out(self):
        new = self.direction - 10 *normalize(self.direction)
        if np.dot(new,normalize(self.direction)) <= 1:
            return
            
        self.direction = new
    
    def step_forward(self,d):
        self.position = self.position + d * normalize(self.direction)

    def step_sideways(self,d):
        self.position = self.position + d * self.tilt

    def fly(self,d):
        self.position = self.position + d * np.array([0,0,1])


class circle():
    def __init__ (self,radius = 100,normal = (0,0,1),control = (1,0,0), position = (0,0,0),show_normal = True):
        self.radius = radius
        self.normal = normalize(normal)
        self.control = normalize(control)
        self.position = np.array(position)
    
    def rotate_roll(self,a):
        self.normal = rotate(self.normal,self.control,a)


    def rotate_pitch(self,a):
        axis = cross(self.normal,self.control)
        self.normal = rotate(self.normal,axis,a)
        self.control = rotate(self.control,axis,a)


    def getmesh(self, n = 7):
        a = np.pi / n
        polygons = []
        pos = self.position
        for x in range(-n,n):
            angle = x * a  
            if x % 2 == 1:
                color = (100, 100, 100)
            else:
                color = (80,80,80)
            w = cross(self.control, self.normal)
            polygons.append(polygon([pos, tuple(self.radius*(w*np.cos(angle) + self.control * np.sin(angle))),tuple(self.radius*(w*np.cos(angle + a) + self.control * np.sin(angle + a)))],color))
        polygons.append(polygon([pos,pos, tuple(100 * self.normal)],(0,255,0)))
        polygons.append(polygon([pos,pos,tuple(100*self.control)], (0,0,255)))
        return polygons


class sphere:
    def __init__ (self,radius = 1,center = (0,0,0), color = (149, 172, 196)):
        self.radius = radius
        self.center = np.array(center)
        self.color = color

    def setposition (self,p):
        self.center = np.array(p)
    
    def setradius (self,r):
        self.radius = r
    
    def getmesh(self, view_point, n = 10):
        polygons = []
        view = normalize(view_point - self.center)
        pos = self.center
        a = np.pi/n
        temp = view + i
        temp2 = project(temp,view)
        normalx = normalize(temp - temp2)
        normaly = cross(view,normalx)
        for x in range(-n,n):
            angle = x * a
            polygons.append(polygon([pos, tuple( pos + self.radius*(normaly*np.cos(angle) + normalx* np.sin(angle))),tuple(pos + self.radius*(normaly*np.cos(angle + a) + normalx* np.sin(angle + a)))],self.color))
        return polygons


"""
state:
0 -> normalx
1 -> normaly
2 -> normalz
3 -> controlx
4 -> controly
5 -> controlz
6 -> ballx
7 -> bally
8 -> ballz
9 -> ballvelocityx
10 -> ballvelocityy
11 -> ballvelocityz
12 -> is_terminal

action:
0 -> pitch+
1 -> pitch-
2 -> roll+
3 -> roll-
4 -> none
"""
        
def reward(state):
    normal = np.array([state[0],state[1],state[2]])
    control = np.array([state[3],state[4],state[5]])
    ball_position = np.array([state[6],state[7],state[8]])
    ball_velocity = np.array([state[9],state[10],state[11]])
    r = length(ball_position - origin)
    temp = -1 * normalize(ball_velocity)
    offset = normalize(ball_position - origin)
    reward = 0
    
    similarity = np.dot(temp,offset)
    if similarity >= 0:
        reward += 5 * np.dot(temp,offset)
    
    
    distance_reward = max(0, 1 - r/100)  
    reward += 2 * distance_reward
    
  
    reward += 0.1  
    
    if r >= 100:
        reward -= 1000
    
    return reward
    

def transition (state,action):
    normal = np.array([state[0],state[1],state[2]])
    control = np.array([state[3],state[4],state[5]])
    ball_position = np.array([state[6],state[7],state[8]])
    ball_velocity = np.array([state[9],state[10],state[11]])
    angle = 0
    newnormal = normal
    newcontrol = control
    newposition = ball_position
    newvelocity = ball_velocity
    is_terminal = False
    if action == 0:
        axis = control
        angle = 0.05
    if action == 1:
        axis = control
        angle = -0.05
    if action == 2:
        axis = cross(normal,control)
        angle = 0.05
    if action == 3:
        axis = cross(normal,control)
        angle = -0.05
    
    if angle != 0:
        newnormal = rotate(normal,axis,angle)
        newcontrol = rotate(control,axis,angle)
        newposition = rotate(ball_position,axis,angle)
        newvelocity = rotate(ball_velocity,axis,angle)
    if np.dot(newnormal,k) <= 0.9:
        newnormal = normal
        newcontrol = control
        newposition = ball_position
        newvelocity = ball_velocity
    projectedgravity = gravity - project(gravity,newnormal)
    newposition = newposition + newvelocity * dt 
    newvelocity = newvelocity + projectedgravity * dt
    newposition = newposition - project(newposition - np.array([0,0,0]), newnormal)
    newvelocity = newvelocity - project(newvelocity, newnormal)
    if length(newposition - np.array([0,0,0])) > 100:
        is_terminal = True
    out = [
    newnormal[0],
    newnormal[1],
    newnormal[2],
    newcontrol[0],
    newcontrol[1],
    newcontrol[2],
    newposition[0],
    newposition[1],
    newposition[2],
    newvelocity[0],
    newvelocity[1],
    newvelocity[2],
    is_terminal
    ]
    return np.array(out)

def randrange (a,b):
    return a + (b-a) * random()

user = viewer(initial_position = (-20,0,100))
world = environment()
platform = circle()
ball = sphere(7)


def printstate(state):
    world.clear()
    normal = np.array([state[0],state[1],state[2]])
    control = np.array([state[3],state[4],state[5]])
    ball_position = np.array([state[6],state[7],state[8]])
    ball_velocity = np.array([state[9],state[10],state[11]])
    platform.normal = normal
    platform.control = control
    ball.center = ball_position + ball.radius/2 * normal
    meshes = platform.getmesh()
    for x in meshes:
        world.add(x)
    polygons = []
    polygons.append(polygon([(0,0,0),(0,0,0),tuple(50 *i)],(255,255,0)))
    polygons.append(polygon([(0,0,0),(0,0,0),tuple(50 *j)],(255,255,0)))
    polygons.append(polygon([(0,0,0),(0,0,0),tuple(50 *k)],(255,255,0)))
    for x in polygons:
     world.add(x)
    for x in ball.getmesh(user.position):
        world.add(x)

    user.renderworld(world)


    
meshes = platform.getmesh()

for x in meshes:
    world.add(x)

polygons = []

polygons.append(polygon([(0,0,0),(0,0,0),tuple(50 *i)],(255,255,0)))
polygons.append(polygon([(0,0,0),(0,0,0),tuple(50 *j)],(255,255,0)))
polygons.append(polygon([(0,0,0),(0,0,0),tuple(50 *k)],(255,255,0)))

for x in polygons:
    world.add(x)





state = np.array([0,0,1,1,0,0,0,0,0,randrange(-10,10),randrange(-10,10),randrange(-10,10),False])


def printpath(path):
    points = []
    axis1 = normalize(cross(platform.normal,platform.control))
    axis2 = normalize(platform.control)
    for p in path:
        vector = p[0] * axis1 + p[1] * axis2
        points.append(vector)
    for x in range(len(points) - 1):
        start = user.renderpoint(points[x])
        end = user.renderpoint(points[x+1])
        if not (np.all(start == 0) or np.all(end == 0)):
            start_transformed = transform(tuple(start))
            end_transformed = transform(tuple(end))
            pygame.draw.line(screen, (255, 255, 255), start_transformed, end_transformed)



def testrun (policy):
    state = np.array([0,0,1,1,0,0,0,0,0, randrange(-10,10),randrange(-10,10),randrange(-10,10),False])
    points = []
    totalrewards = 0
    loop = True
    while loop:
        if state[len(state) - 1] :
            loop = False
        normal = np.array([state[0],state[1],state[2]])
        control = np.array([state[3],state[4],state[5]])
        ball_position = np.array([state[6],state[7],state[8]])
        ball_velocity = np.array([state[9],state[10],state[11]])
        axis1 = normalize(cross(normal,control))
        axis2 = normalize(control)
        xtemp = np.dot(ball_position,axis1)
        ytemp = np.dot(ball_position,axis2)
        points.append(np.array([xtemp,ytemp]))
        if len(points) > 100:
            points.pop(0)
        printstate(state)
        printpath(points)
        pygame.draw.line(screen,(255,128,0),transform(tuple(user.renderpoint(ball_position))),transform(tuple(user.renderpoint(ball_position + ball_velocity))))
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEMOTION:
                dx, dy = event.rel
                rotation_factor = 0.005 
                user.rotate_horizontal(-dx * rotation_factor)
                user.rotate_vertical(dy * rotation_factor)

        action = policy.getaction(state)[0]
        keys = pygame.key.get_pressed()
        # action = 4
        # if keys[K_i]:
        #     action = 0
        # if keys[K_k]:
        #     action = 1
        # if keys[K_j]:
        #     action = 2
        # if keys[K_l]:
        #     action = 3
        if keys[K_w]:
            user.step_forward(10)
        if keys[K_a]:
            user.step_sideways(-10)
        if keys[K_s]:
            user.step_forward(-10)
        if keys[K_d]:
            user.step_sideways(10)
        if keys[K_COMMA]:
            user.zoom_in()
        if keys[K_PERIOD]:
            user.zoom_out()
        if keys[K_UP]:
            user.rotate_vertical(-0.04)
        if keys[K_DOWN]:
            user.rotate_vertical(0.04)
        if keys[K_LEFT]:
            user.rotate_horizontal(0.04)
        if keys[K_RIGHT]:
            user.rotate_horizontal(-0.04)
        if keys[K_SPACE]:
            user.fly(10)
        if keys[K_LSHIFT]:
            user.fly(-10)
        state = transition(state,action)
        totalrewards += reward(state)
        
        draw_text(screen,"zoom : " + str(round(length(user.direction))), (255,0,0), (1350,50),23)
        draw_text(screen, str((np.round(user.position,decimals = 1))),(255,0,0), (1350,100),23)
        draw_text(screen, "total rewards: " + str(totalrewards),(255,0,0), (1350,150),23)
        
        pygame.display.update()
        
        clock.tick(60)
            
 

policy = agent()
policy.loadparameters("recordpolicy.pth","recordvalue.pth")
testrun(policy)



n = 0    

while True:
    print("trial number : " + str(n))
    if n % 50 == 0:
        testrun(policy)
    n += 1
    done = False
    for _ in range (20):
        state = np.array([0,0,1,1,0,0,0,0,0,2* randrange(-10,10),2 * randrange(-10,10),2 * randrange(-10,10),False])
        while not done:
            action,prob,val = policy.getaction(state)
            nextstate = transition(state,action)
            reward_ = reward(nextstate)
            if state[len(state) - 1]:
                done = True
            policy.storememory(state,action,prob,val,reward_,)
            state = nextstate
        policy.endtrajectory()
    policy.train(20)
