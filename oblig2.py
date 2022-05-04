# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.io as sio
# Dette virker med versjon 7 MAT-filer
data = sio.loadmat(r"C:\Users\Anders sin PC\Documents\UIO\V22\MEK1100\oblig2\data.mat")
x = data.get("x")
y = data.get("y")
u = data.get("u")
v = data.get("v")
xit = data.get("xit")
yit = data.get("yit")




def plt_skilleflate():
    plt.plot(xit[0], yit[0], color="black") 
    
def plt_rektangler():
    reks = [[[x[0,35], y[160,0]], [x[0,70], y[170,0]]]
        , [[x[0,35], y[85,0]], [x[0,70], y[100,0]]]
        , [[x[0,35], y[50,0]], [x[0,70], y[60,0]]]]
    
    colours = [["red", "black"], ["blue", "green"]]
    for rek in reks:
        for i in [0,1]:
            
            plt.plot([rek[0][0],rek[1][0]], [rek[i][1], rek[i][1]], color = colours[i][0])
            
            plt.plot([rek[i][0],rek[i][0]], [rek[0][1], rek[1][1]], color = colours[i][1])

def oppgave_a():
    # Sjekk at x,y,u,v har riktig dimensjoner og str
    matrices = ["x", "y", "u", "v"]
    for mat in matrices:
        # Since these are numy objects i can use their atttributes to get metadata
        # The first (and all) list in matrix should be 194 elements long
        statement_xlength = f"{mat}.shape[1]==194"
        # The number of lists, i.e. length of matrix, should be 201 elements long
        statement_ylength = f"{mat}.shape[0]==201"
        # Check that their products is 38994
        statement_product = f"{mat}.size==38994"
        # Using python eval func to iterate and check
        print(f"matrix {mat} has correct dimensions and size: {eval(statement_xlength) and eval(statement_ylength) and eval(statement_product)}")
               
    # Sjekk at vektorene har riktig størelse i x-retning
    vectors = ["xit", "yit"]
    for vec in vectors:
        statement_xlength = f"{vec}.shape[1]==194"
        print(f"vector {vec} has correct x-length: {eval(statement_xlength)}")


    # Sjekk at mellomrommet mellom grid-punkter er alle 0.5 mm
    # Kan bruke numpy's diff() funksjon, med forskjellig axis-valg for å sjekke intervallene mellom elementene 
    print(f"x matrix difference between gridpoints is 0.5mm: {(np.diff(x, axis=-1)==0.5).all()}") 
    print(f"y matrix difference between gridpoints is 0.5mm: {(np.diff(y, axis=-2)==0.5).all()}") 
    
    # Sjekk at y-kordinatene spenner hee diameteren til røret
    # Diameteren er 10 cm (2*5cm radius)
    diam = 100
    print(f"y interval spans diameter of 100mm: {np.max(y)-np.min(y)==diam}")
    
def oppgave_b():
    
    z = np.sqrt(u**2 + v**2)
    
    limit = 1000
    q = np.where(z>=limit, z, limit)
    plt.contour(x, y, q, levels=1000, cmap= "RdBu")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt_skilleflate()
    plt.colorbar()
    plt_title = "Vectorfield Speed, gass"
    plt.title(plt_title)
    plt.show()
    plt.clf()
    
    limit = 1000
    p = np.where(z<limit, z, limit)
    plt.contour(x, y, p, levels=1000, cmap= "RdBu")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.plot(xit[0], yit[0], color="black")
    plt.colorbar()
    plt_title = "Vectorfield Speed, liquid"
    plt.title(plt_title)
    plt.show()
    plt.clf()
    
    plt.contour(x, y, z, levels=1000, cmap= "RdBu")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.plot(xit[0], yit[0], color="black")
    plt.colorbar()
    plt_title = "Vectorfield Speed"
    plt.title(plt_title)
    plt.show()
    plt.clf()
    

def oppgave_c():
    
    
    
    arrow_interval = 9
    
    plt.quiver(x[0:len(y):arrow_interval, 0:len(x):arrow_interval],
               y[0:len(y):arrow_interval, 0:len(x):arrow_interval],
               u[0:len(y):arrow_interval, 0:len(x):arrow_interval],
               v[0:len(y):arrow_interval, 0:len(x):arrow_interval],
               headlength=3,
               headwidth=2)
    
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt_skilleflate()
    plt_rektangler()
    plt_title="Vectorfield"
    plt.title(plt_title)
    plt.show()
    plt.clf()
    


def oppgave_d():
    
    # The divergence theorem relates the divergence of the colume to the integral of the 
    # surface of the volume. We can use this here, although of course we only have 2 dimension, 
    # so it will be a surface -> line integral relation.
    
    # Surface integral
    diff_u = np.diff(u, axis=1)*0.5 # multiplied by dy, which is 0.5mm in our dataset.
    #add zero vector to fill grid (these zeroes do not influence the divergence calculations)
    diff_u = np.append(diff_u,np.zeros((201,1)), axis=1)

    diff_v = np.diff(v, axis=0)*0.5 # multiplied by dx, which is 0.5mm in our dataset
    #add zero vector to fill grid (these zeroes do not influence the divergence calculations)
    diff_v = np.append(diff_v,np.zeros((1,194)), axis=0)
    div = diff_v + diff_u
    surf_sum = np.sum(div)

    # Lineintegral
    sum_b = np.sum(v[0]*-0.5)
    sum_t = np.sum(v[200]*0.5)
    sum_l = np.sum(u[:,0]*-0.5)
    sum_r = np.sum(u[:,193]*0.5)
    
    line_sum = sum_b+sum_l+sum_r+sum_t

    
    print("\nResults from two divergence caluclations (LHS and RHS, Gauss' theorem):")
    print(f"Surface integral: {surf_sum}")
    print(f"Line integral:    {line_sum}")
    print(f"Difference: {abs(surf_sum-line_sum)}")
    
    
    # # Conttour plot for curl


    # I'm limiting the scope of the divergence in the plot to make the contours more distinct
    p = np.where(div>-30, div, -30)
    p = np.where(p<100, p, 100)
    plt.contourf(x, y, p, cmap="rainbow")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt_skilleflate()
    plt_rektangler()
    plt.colorbar()
    plt_title = "Divergence of vectorfield"
    plt.title(plt_title)
    plt.show()
    plt.clf()
    
def oppgave_e():
    
    # I will calculate both sides of greens theorem seperately. First the surface integral,
    # then the line integral along the perimiter. 
    
    # Surface integral:
        
    # Calculating differences in vectorfield along perpendiculer axis, 
    # then summing up over all points 
    diff_u = np.diff(u, axis=0)*0.5 # multiplied by dy, which is 0.5mm in our dataset.
    #add zero vector to fill grid (these zeroes do not influence the curl calculations)
    diff_u = np.append(diff_u,np.zeros((1,194)), axis=0)  

    
    diff_v = np.diff(v, axis=1)*0.5 # multiplied by dx, which is 0.5mm in our dataset
    #add zero vector to fill grid (these zeroes do not influence the curl calculations)
    diff_v = np.append(diff_v,np.zeros((201,1)), axis=1)

    
    curl = diff_v - diff_u
    surf_sum = np.sum(curl)
    
    
    # Line integral:
    # All we need to do here is sum up the vectorfield along the borders of the domain. 
    # Since the domain (i.e. our surface) is a rectangle, this process becomes incredible simple:
    # When we are 
    
    # We have to proceed in a positive direction, i.e. following the right hand rule
    # first the bottom side:
    sum_b = np.sum(u[0]*0.5)
    # the the top side, but now we are going in the negative direction (following right hand rule):
    sum_t = np.sum(u[200]*-0.5)
    # sum over left side
    sum_l = np.sum(v[:,0]*-0.5)
    # sum over right side
    sum_r = np.sum(v[:,193]*0.5)
    
    line_sum = sum_b+sum_l+sum_r+sum_t


    # The values are unbelivably close!
    print("\nResults from two curl caluclations (LHS and RHS, Grens theorem):")
    print(f"Surface integral: {surf_sum}")
    print(f"Line integral:    {line_sum}")
    print(f"Difference: {abs(surf_sum-line_sum)}")
    
    
    # Conttour plot for curl
    
    limit = 70
    p = np.where(curl>-limit, curl, -limit)
    p = np.where(p<limit, p, limit)
    plt.contourf(x, y, p)
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt_skilleflate()
    plt_rektangler()
    plt.colorbar()
    plt_title = "Curl of vectorfield"
    plt.title(plt_title)
    plt.show()
    plt.clf()
    
    
    arrow_interval = 3
    plt.contour(x[0:len(y):arrow_interval, 0:len(x):arrow_interval],
           y[0:len(y):arrow_interval, 0:len(x):arrow_interval],
           curl[0:len(y):arrow_interval, 0:len(x):arrow_interval],
           levels=100)
    plt_skilleflate()
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.title("Vectorfield streamlines")
    plt.show()
    plt.clf()

def oppgave_f():

    
    # Line integral:
    # All we need to do here is sum up the vectorfield along the borders of the rectangles. 
    
    reks = [
        [[35,160], [70,170]]
    , [[35,85], [70,100]]
    , [[35,50], [70,60]]
    ]
    
    
    # Surface integral
    diff_u = np.diff(u, axis=0)*0.5 # multiplied by dy, which is 0.5mm in our dataset.
    #add zero vector to fill grid (these zeroes do not influence the curl calculations)
    diff_u = np.append(diff_u,np.zeros((1,194)), axis=0)      
    diff_v = np.diff(v, axis=1)*0.5 # multiplied by dx, which is 0.5mm in our dataset
    #add zero vector to fill grid (these zeroes do not influence the curl calculations)
    diff_v = np.append(diff_v,np.zeros((201,1)), axis=1)
    curl = diff_v - diff_u
    
    
    
    for ind, rek in enumerate(reks):
        
        # We have to proceed in a positive direction, i.e. following the right hand rule
        # first the bottom side:
        sum_b = np.sum(u[rek[0][1], rek[0][0]:rek[1][0]]*0.5)
        # the the top side, but now we are going in the negative direction (following right hand rule):
        sum_t = np.sum(u[rek[1][1],rek[0][0]:rek[1][0]]*-0.5)
        # sum over left side
        sum_l = np.sum(v[rek[0][1]:rek[1][1],rek[0][0]]*-0.5)
        # sum over right side
        sum_r = np.sum(v[rek[0][1]:rek[1][1],rek[1][0]]*0.5)
        
        line_sum = sum_b+sum_l+sum_r+sum_t
        surf_sum = np.sum(curl[rek[0][1]:rek[1][1],rek[0][0]:rek[1][0]])
    
        print(f"\nLine integral   : Curl of rectangle {ind}: {line_sum}")
        print(f"Surface integral: Curl of rectangle {ind}: {surf_sum}")


def oppgave_g():
    
    # Line integral:
    # All we need to do here is sum up the vectorfield along the borders of the rectangles. 
    
    reks = [
        [[35,160], [70,170]]
    , [[35,85], [70,100]]
    , [[35,50], [70,60]]
    ]
    
    
    # Surface integral
    diff_u = np.diff(u, axis=1)*0.5 # multiplied by dy, which is 0.5mm in our dataset.
    #add zero vector to fill grid (these zeroes do not influence the divergence calculations)
    diff_u = np.append(diff_u,np.zeros((201,1)), axis=1)

    diff_v = np.diff(v, axis=0)*0.5 # multiplied by dx, which is 0.5mm in our dataset
    #add zero vector to fill grid (these zeroes do not influence the divergence calculations)
    diff_v = np.append(diff_v,np.zeros((1,194)), axis=0)
    div = diff_v + diff_u
    
    
    
    for ind, rek in enumerate(reks):
        print(f"\nRectangle {ind}:")
        
        # We have to proceed in a positive direction, i.e. following the right hand rule
        # first the bottom side:
        sum_b = np.sum(v[rek[0][1], rek[0][0]:rek[1][0]]*-0.5)
        print(f"Divergence bottom: {sum_b}")
        # the the top side, but now we are going in the negative direction (following right hand rule):
        sum_t = np.sum(v[rek[1][1],rek[0][0]:rek[1][0]]*0.5)
        print(f"Divergence top:    {sum_t}")
        # sum over left side
        sum_l = np.sum(u[rek[0][1]:rek[1][1],rek[0][0]]*-0.5)
        print(f"Divergence left:   {sum_l}")
        # sum over right side
        sum_r = np.sum(u[rek[0][1]:rek[1][1],rek[1][0]]*0.5)
        print(f"Divergence right:  {sum_r}")
        
        line_sum = sum_b+sum_l+sum_r+sum_t
        
        
        surf_sum = np.sum(div[rek[0][1]:rek[1][1],rek[0][0]:rek[1][0]])
    
        print(f"Line integral   : Divergence of rectangle {ind}: {line_sum}")
        print(f"Surface integral: Divergence of rectangle {ind}: {surf_sum}")

        
    


if __name__ == "__main__":
    
    tasks = ["a", "b", "c", "d", "e", "f", "g"]
    
    for task in tasks:
        string = "\noppgave_"+task+"()"
        print(string)
        eval(string)




# Kjøreeksempel c:\>python oblig2.py
# =============================================================================
# oppgave_a()
# matrix x has correct dimensions and size: True
# matrix y has correct dimensions and size: True
# matrix u has correct dimensions and size: True
# matrix v has correct dimensions and size: True
# vector xit has correct x-length: True
# vector yit has correct x-length: True
# x matrix difference between gridpoints is 0.5mm: True
# y matrix difference between gridpoints is 0.5mm: True
# y interval spans diameter of 100mm: True
# 
# oppgave_b()
# Locator attempting to generate 1001 ticks ([0.0, ..., 1000.0]), which exceeds Locator.MAXTICKS (1000).
# 
# oppgave_c()
# 
# oppgave_d()
# 
# Results from two divergence caluclations (LHS and RHS, Gauss' theorem):
# Surface integral: -11740.44211621365
# Line integral:    -11740.442116213635
# Difference: 1.4551915228366852e-11
# 
# oppgave_e()
# 
# Results from two curl caluclations (LHS and RHS, Grens theorem):
# Surface integral: -25997.019444898324
# Line integral:    -25997.019444898327
# Difference: 3.637978807091713e-12
# 
# oppgave_f()
# 
# Line integral   : Curl of rectangle 0: 3095.8376854420057
# Surface integral: Curl of rectangle 0: 3095.837685441991
# 
# Line integral   : Curl of rectangle 1: -58971.02035274107
# Surface integral: Curl of rectangle 1: -58971.02035274108
# 
# Line integral   : Curl of rectangle 2: -45.08533659914792
# Surface integral: Curl of rectangle 2: -45.085336599148924
# 
# oppgave_g()
# 
# Rectangle 0:
# Divergence bottom: 1594.8147947210718
# Divergence top:    -2032.1499329641288
# Divergence left:   -19196.992492825088
# Divergence right:  19655.88502439653
# Line integral   : Divergence of rectangle 0: 21.557393328386297
# Surface integral: Divergence of rectangle 0: 21.5573933283859
# 
# Rectangle 1:
# Divergence bottom: -5131.173666473514
# Divergence top:    -3688.42376249757
# Divergence left:   -12744.859067516707
# Divergence right:  13682.855773041552
# Line integral   : Divergence of rectangle 1: -7881.6007234462395
# Surface integral: Divergence of rectangle 1: -7881.60072344624
# 
# Rectangle 2:
# Divergence bottom: -242.52548830520567
# Divergence top:    291.53512698003465
# Divergence left:   -1560.5800387411343
# Divergence right:  1404.3067306552778
# Line integral   : Divergence of rectangle 2: -107.26366941102765
# Surface integral: Divergence of rectangle 2: -107.26366941102773
# <Figure size 432x288 with 0 Axes>
# 
# 
# =============================================================================
