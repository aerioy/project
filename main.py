
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

i = np.array([1,0,0])
j = np.array([0,1,0])
k = np.array([0,0,1])

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


class platform_mesh():
    def __init__ (self,radius = 100,edge_width = 20,normal = (0,0,1),control = (1,0,0)):
        self.radius = radius
        self.edge_width = edge_width
        self.normal = normalize(normal)
        self.control = normalize(control)
    
    def rotate_roll(self,a):
        normal = cross(self.normal,self.control)
        newdirection = self.normal * np.cos(a) + normal * np.sin(a)
        self.normal = normalize(newdirection)


    def rotate_pitch(self,a):
        normal = cross(self.normal,self.control)
        newdirection = self.normal * np.cos(a) + self.control * np.sin(a)
        self.normal = normalize(newdirection)
        self.control = cross(normal,self.normal)


    def getmesh(self):
        a = np.pi / 10
        polygons = []
        for x in range(-10,10):
            angle = x * a  
            if x % 2 == 1:
                color = (50,50,50)
            else:
                color = (100,100,100)
            w = cross(self.control, self.normal)
            polygons.append(polygon([(0,0,0), tuple(self.radius*(w*np.cos(angle) + self.control * np.sin(angle))),tuple(self.radius*(w*np.cos(angle + a) + self.control * np.sin(angle + a)))],color))
        polygons.append(polygon([(0,0,0),(0,0,0), tuple(100 * self.normal)],(0,255,0)))
        polygons.append(polygon([(0,0,0),(0,0,0),tuple(100*self.control)], (0,0,255)))
        polygons.append(polygon([(0,0,0),(0,0,0),tuple(50 *i)],(255,255,0)))
        polygons.append(polygon([(0,0,0),(0,0,0),tuple(50 *j)],(255,255,0)))
        polygons.append(polygon([(0,0,0),(0,0,0),tuple(50 *k)],(255,255,0)))
        return polygons






user = viewer(initial_position = (-20,0,100))
world = environment()
platform = platform_mesh()

meshes = platform.getmesh()

for x in meshes:
    world.add(x)










while True:
    world.clear()
    meshes = platform.getmesh()
    for x in meshes:
        world.add(x)
    user.renderworld(world)
    for event in pygame.event.get():
         if event.type == QUIT:
            pygame.quit()
            sys.exit()
        
    keys = pygame.key.get_pressed()
      
    if keys[K_w]:
        user.step_forward(10)
    if keys[K_a]:
        user.step_sideways(-10)
    if keys[K_s]:
        user.step_forward(-10)
    if keys[K_d]:
        user.step_sideways(10)
    if keys[K_j]:
        platform.rotate_roll(0.05)
    if keys[K_l]:
        platform.rotate_roll(-0.05)
    if keys[K_i]:
        platform.rotate_pitch(0.05)
    if keys[K_k]:
        platform.rotate_pitch(-0.05)
    if keys[K_COMMA]:
        user.zoom_in()
    if keys[K_PERIOD]:
        user.zoom_out()
    if keys[K_UP]:
        user.rotate_vertical(-0.02)
    if keys[K_DOWN]:
        user.rotate_vertical(0.02)
    if keys[K_LEFT]:
        user.rotate_horizontal(0.02)
    if keys[K_RIGHT]:
        user.rotate_horizontal(-0.02)
    if keys[K_SPACE]:
        user.fly(10)
    if keys[K_LSHIFT]:
        user.fly(-10)
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    draw_text(screen,"zoom : " + str(round(length(user.direction))), (255,0,0), (1350,50),23)
    draw_text(screen, str((np.round(user.position,decimals = 1))),(255,0,0), (1350,100),23)

    pygame.display.update()
    
    clock.tick(60)
        
 

