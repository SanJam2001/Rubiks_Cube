#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


#white=0, orange=1, yellow=2, red=3, green=4, blue=5


# In[3]:


x=np.zeros(9,dtype = int).reshape(3,3)
face_front = x
face_down = np.full_like(x,1)
face_back = np.full_like(x,2)
face_up = np.full_like(x,3)
face_left = np.full_like(x,4)
face_right = np.full_like(x,5)


# In[4]:


cube_solved=np.array([face_front,face_down,face_back,face_up,face_left,face_right])
print(cube_solved)


# In[5]:


#take the right handed coordinate system
#positive x,y and z axes pass through the right, up and front faces respectively
#rot_x are rotations about x axis and angular velocity is supposed to be in +ve x direction
#rot_y are rotations about y axis and angular velocity is supposed to be in +ve y direction
#rot_z are rotations about z axis and angular velocity is supposed to be in +ve z direction
"""
numbering of individual blocks is done as follows:
take the cube in your hand with face_front facing you.
number rows and columns from top to bottom and left to right respectively.
flip the cube in the sense opposite to rot_x such that face_bottom is facong you. follow the same pattern to number.
flip the cube two more times to number face_back and face_up.
turn the cube in the sense of rot_y and number face_left.
turn the cube in the sense opposite to rot_y and nymber face_right.
"""


# In[6]:


def rot_x(n,col,cube_prev):
    #cube_prev=the arrangement of cube on which we wish to perform the rotation
    #n=no._of_rotations(<=4)
    #col=column_no.(=0,1,2)
    cube_initial=cube_prev.copy()
    cube_initial1=cube_prev.copy()
    l=[0,1,2,3]
    for i in range(0,3):
        l0=[i,col]
        cube_initial1[0,i,col] = cube_initial[l[4-n],i,col]
        cube_initial1[1,i,col] = cube_initial[l[1-n],i,col]
        cube_initial1[2,i,col] = cube_initial[l[2-n],i,col]
        cube_initial1[3,i,col] = cube_initial[l[3-n],i,col]
    l1=[(0,1),(1,2),(2,1),(1,0)]
    l2=[(0,0),(0,2),(2,2),(2,0)]
    if col==2:
        for i in range(0,4):
            cube_initial1[5][l1[i]] = cube_initial[5][l1[i-4+n]]
            cube_initial1[5][l2[i]] = cube_initial[5][l2[i-4+n]]
    if col==0:
        for i in range(0,4):
            cube_initial1[4][l1[i]] = cube_initial[4][l1[i-n]]
            cube_initial1[4][l2[i]] = cube_initial[4][l2[i-n]]
    return cube_initial1


# In[7]:


def rot_y1(row,cube_prev):
    #cube_prev=the arrangement of cube on which we wish to perform the rotation.
    #row=row_no.(=0,1,2)
    cube_initial=cube_prev.copy()
    cube_initial1=cube_prev.copy()
    l=[2,1,0]
    for i in range(0,3):
        cube_initial1[0,row,i]=cube_initial[4,row,i]
        cube_initial1[5,row,i]=cube_initial[0,row,i]
        cube_initial1[2,l[row],l[i]]=cube_initial[5,row,i]
        cube_initial1[4,row,i]=cube_initial[2,l[row],l[i]]
    l1=[(0,1),(1,2),(2,1),(1,0)]
    l2=[(0,0),(0,2),(2,2),(2,0)]
    if row==0:
        for i in range(0,4):
            cube_initial1[3][l1[i]] = cube_initial[3][l1[i-3]]
            cube_initial1[3][l2[i]] = cube_initial[3][l2[i-3]]
    if row==2:
        for i in range(0,4):
            cube_initial1[1][l1[i]] = cube_initial[1][l1[i-1]]
            cube_initial1[1][l2[i]] = cube_initial[1][l2[i-1]]
    return cube_initial1

def rot_y(n,row,cube_prev):
    #n=no. of rotations
    cube=cube_prev
    for i in range(1,n+1):
        a=rot_y1(row,cube)
        cube=a
    return cube


# In[8]:


def rot_z1(layer,cube_prev):
    #cube_prev=the arrangement of cube on which we wish to perform the rotation
    #layer=layer_no.(=0,1,2)
    cube_initial=cube_prev.copy()
    cube_initial1=cube_prev.copy()
    l1=[2,1,0]
    for i in range(0,3):
        cube_initial1[3,layer,i] = cube_initial[5,i,l1[layer]]
        cube_initial1[4,i,layer] = cube_initial[3,layer,l1[i]]
        cube_initial1[1,l1[layer],i] = cube_initial[4,i,layer]
        cube_initial1[5,i,l1[layer]] = cube_initial[1,l1[layer],l1[i]]
    l2=[(0,1),(1,2),(2,1),(1,0)]
    l3=[(0,0),(0,2),(2,2),(2,0)]
    if layer==0:
        for i in range(0,4):
            cube_initial1[2][l2[i]] = cube_initial[2][l2[-1-i]]
            cube_initial1[2][l3[i]] = cube_initial[2][l3[-1-i]]
    if layer==2:
        for i in range(0,4):
            cube_initial1[0][l2[i]] = cube_initial[0][l2[i-3]]
            cube_initial1[0][l3[i]] = cube_initial[0][l3[i-3]]
    return cube_initial1

def rot_z(n,layer,cube_prev):
    cube=cube_prev
    for i in range(1,n+1):
        a=rot_z1(layer,cube)
        cube=a
        return cube


# In[9]:


import random


# In[10]:


#this function jumbles up the cube for you
def jumble(n):
    l1=[0,1,2]
    l2=[1,2,3,4]
    cube_prev=cube_solved.copy()
    for i in range(0,n):
        a=rot_x(random.choice(l2),random.choice(l1),cube_prev)
        b=rot_y(random.choice(l2),random.choice(l1),a)
        c=rot_z(random.choice(l2),random.choice(l1),b)
        cube_prev=c
    return cube_prev


# In[11]:


def trial1(n,k):
    a= jumble(k)
    count=0
    for i in range(0,n):
        count=count+1
        for j in range(0,3):
            b=rot_x(1,j,a)
            c=rot_y(1,j,b)
            a=rot_z(1,j,c)
        if np.array_equal(a,cube_solved)==True:
            return count
    return "loop ended"


# In[12]:


def track1(a,n):
    l=[]
    cube=cube_solved.copy()
    cube[a]=6
    for i in range(0,n):
        for j in range(0,3):
            b=rot_x(1,j,cube)
            c=rot_y(1,j,b)
            cube=rot_z(1,j,c)
        for m in range(0,6):
            for n in range(0,3):
                for o in range(0,3):
                    if cube[m,n,o]==6:
                        l1=np.array([m,n,o])
                        l.append(l1)
    count=0
    for i in range(1,len(l)):
        count=count+1
        if np.array_equal(l[i],l[0])==True:
            return count,l
    return l


# In[13]:


l=[]
for i in range(0,6):
    for j in range(0,3):
        for k in range(0,3):
            l.append(track1((i,j,k),25)[0])
print(l)


# In[14]:


#so it is clear that after 40 cycles the cube will repeat. hence, if trail1(n,k) has to work, it will work for n=41


# In[15]:


import math as m
m.factorial(9)


# In[16]:


def lcm(l):
    arr = np.zeros(len(l),dtype=object)

    l_reduced=[]
    l_a=sorted(l)
    for i in l_a:
        l_reduced.append(i)
        for j in l:
            if j==i:
                l_a.remove(j)

    for i in range(0,len(l_reduced)):
        x=l_reduced[i]
        l1=[]
        for j in range(1,x+1):
            if x%j==0:
                l1.append(j)
        arr[i]=l1
    l_common = []
    for i in arr[0]:
        count=0
        for j in range(1,len(l_reduced)):
            for k in arr[j]:
                if i==k:
                    count=count+1
        if count == len(l_reduced)-1:
            l_common.append(i)
    hcf = max(l_common)

    product=1
    for i in l_reduced:
        product = product*i
    lcm = product/hcf
    return lcm


# In[40]:


from itertools import permutations
l3=[[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]
per=permutations(l3)
l_p = list(per)
def alg(n,cube_prev):
    l1=[rot_x,rot_y,rot_z]
    l2=[0,1,2]
    cube=np.array(cube_prev).copy()
    l_n = l_p[n]
    for i in range(0,9):
        a=l1[l_n[i][0]](1,l_n[i][1],cube)
        cube=a
    return np.array(cube),l_n


# In[18]:


def track(a,n,n1):
    # a = piece you wish to track
    # n = number of cycles of the algorithm you want
    # n1 = number to specify the algorithm
    l=[]
    cube=cube_solved.copy()
    cube[a]=6
    for i in range(0,n):
        cube_new=alg(n1,cube)[0]
        cube=cube_new
        for m in range(0,6):
            for n in range(0,3):
                for o in range(0,3):
                    if cube[m,n,o]==6:
                        l1=np.array([m,n,o])
                        l.append(l1)
    count=0
    for i in range(1,len(l)):
        count=count+1
        if np.array_equal(l[i],l[0])==True:
            return count,l
    return l


# In[19]:


def period(alg_no):
    l=[]
    for i in range(0,6):
        for j in range(0,3):
            for k in range(0,3):
                l.append(track((i,j,k),50,alg_no)[0])
    return lcm(l)


# In[20]:


import matplotlib.pyplot as plt


# In[21]:


x=list(np.arange(0,1000,1))
y=[]
for i in x:
    y.append(period(i))
plt.plot(x,y)
plt.show()


# In[22]:


random_cube = jumble(121)
print(random_cube)


# In[77]:


def plot_center(n1,cube):
    cube_dash = np.array(cube.copy())
    y=[]
    for i in range(0,100):
        cube_dash=alg(n1,cube_dash)[0]
        y.append(np.array(cube_dash)[0][1][1])
    x = list(np.arange(1,101))
    plt.plot(x,y)
    plt.grid()
    plt.show()


# In[87]:


plot_center(3,random_cube)


# In[ ]:





# In[ ]:
