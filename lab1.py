# -*- coding: utf-8 -*-
"""
This file is to introduce basics for Python and Numpy. This file is tested with 
Python verison 3.6.2. This program is extracted from below documents. For original
documents in more thorough details, please see below documents, which are the
original sources of this file.

1. Python : https://docs.python.org/3/tutorial/
2. Numpy : https://numpy.org/doc/stable/user/quickstart.html
3. Python and Numpy Tutorial : https://cs231n.github.io/python-numpy-tutorial/

Made by Min-sung Koh.
"""
# -------------------- Basics for Python -------------------------------------
#---------- 1. Arithemtic operations
a = 11                          # sets 'a' to a int value
print(a / 3)                    # does the calculation 11/3 with the answer as a float (3.6667)
print(a // 3)                   # 11/3 but answer is a int (3)
print(a % 3)                    # modulus operator (2)
print(a * 2)                    # multiples 11 and 2 (22)
print(a ** 2)                   # power function, 11^2, (121)
print('------------')

a += 1                          # adds 1 to the initial value (12)
print(a)                        
a *= 2                          # same as a = a * 2 (24)
print(a)
print('------------')

print(a, type(a))               # prints both the value of a and the data type
print('------------')

print(a, a // 3, a ** 2)        # prints three different answers (24)(8)(576)
print('------------')

#---------- 2. Handling strings
str1 = 'Python'                                                 # single quotes are used for string
str2 = "Numpy"                                                  # or double quotes are also O.K.
print(str1 + ' ' + str2)                                        # prints both strings with a space in between
print(len(str1), len(str2))                                     # prints the length of each string
print('------------')

print(f'This document is to introduce {str1} 3 and {str2}')     # a f-string is to put the other two strings in place
print(f"This document is to introduce {str1} 3 and {str2}")     # showing that both single and double quotes work
str3 = '{} {} and {}'.format(str1,3,str2)                       # string formatting, assigns 'str3' what is in the parentheses
print(str3)                                                     # prints out the above
print('{} {} and {}'.format(str1,3,str2))                       # another way of doing the one above
print('------------')

str4 = 'it is a good day!'                                      # assigning
print(str4.capitalize())                                        # using methods, this capitalizes the first char of the first string
print(str4.upper())                                             # uppercases the whole string
print(str4.replace('a', '(aaa)'))                               # replaces an 'a' with 'aaa'
str5 = '        I hope you guys will enjoy this course!!!         ' 
print(str5.strip())                                             # leading and trailing whitespace is removed 
print('------------')

#---------- 3. Container types such as lists, dictionaries, sets, and tuples
# Python has built-in container types: lists, dictionaries, sets, and tuples.

#----------- 3.1 Lists ----
squares = [1, 4, 9, 16, 25]     # list assigned to 'squares'
print(squares, squares[3])      # prints the whole list along with whats in index 3 (1 4 9 16 25 16)
print(squares[-1])              # last element of the list
print(squares[-3])              # third from last element
print('------------')

squares[2] = 'what?'            # at index 2, that value is replaced with 'what?'
print(squares)                  
print(squares[2], squares[0])   # prints what is at index 2 and 0 (what?)(1)
print('------------')

squares.append('Next?')         # at the end of the list, add 'Next?'
print(squares)                  
squares.pop()                   # removes and returns the last element from the list
print(squares) 
squares.append(36)              # add '36' at the end of the list
print(squares)  
print('------------')

# In Python, "slicing" is available, where "sliceing" means sublists in a list.
nums = list(range(5))           # creates a list 'nums' with elements from 0 to 4 using the range function 
print(nums)                     
print(nums[2:4])                # prints a sublist from the main one, 'nums', and only prints from index 2(inclusive) to 4(exclusive)
print(nums[2:])                 # sublist from index 2 to the end
print(nums[:2])                 # sublist from beginning to index 2
print(nums[:])                  # prints whole list
print(nums[:-1])                # sublist from beginning to the second to last
nums[2:4] = [8, 9]              # replaces index 2 and 4 with 8 and 9
print(nums)                     
print('------------')

# A loop (e.g., "for" statement) can be made using a list as below:
EECourses = ['EENG 320', 'EENG 321', 'EENG 420', 'EENG 440']    # EECourses assigned a list of strings
for indx, EEClass in enumerate(EECourses):                      # a for loop that iterates over the elements of EECourses, enumerate gets the index(indx) and the value(EEClass) 
    print(f'#{indx} class: {EEClass}')                          # prints each iterating
print('------------')
    
# "list comprehension" is a very useful way with iteration loops as below, where
# you want to calculate squares:
nums = [0, 1, 2, 3, 4]
squares = []                                                    # initilizes an empty list
for x in nums:                                                  # variable 'x' in sequence of 'nums'
    squares.append(x ** 2)                                      # appends the square of the current element (x) to the 'squares' list
print(squares)  
# You can make this code simpler using a list comprehension:
nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]                                # assigning 'squares' the for loop directly rather than making a empty list and assigning values later
print(squares)

# List comprehensions can also contain conditions:
nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]             # a for loop and if statment, if no remainder, then even
print(even_squares)
print('------------')

#----------- 3.2 Dictionaries ----
# A dictionary is defined with curly brackets (i.e., {}) ant it stores 
# (key, value) pairs. You can use it like this:
EECoursesQuarters = {'EENG 320': 'Fall quarter', 'EENG 321': 'Spring quarter', \
                     'EENG 420': 'Winter quarter', 'EENG 440': 'Spring quarter'}
EECoursesCredits = {'EENG 420': 5, 'EENG 440': 5, 'EENG 490A': 2, 'EENG 490B': 3}

print(f'EENG 420 having {EECoursesCredits["EENG 420"]} credits will be offered in {EECoursesQuarters["EENG 420"]}')   
print('EENG 490A' in EECoursesCredits)   
print('------------')

EECoursesQuarters['EENG 209'] = 'Fall Quater'   
print(EECoursesQuarters)      
print('------------')

#print(EECoursesQuarters['EENG 360'])  # KeyError: ''EENG 360'' not a key of the dictionarry, EECoursesQuarters 
print('------------')

print(EECoursesCredits)
print(EECoursesCredits.get('EENG 360', 'N/A'))  # Get an element with a default; prints "N/A"
print('------------')

print(EECoursesQuarters)
del EECoursesQuarters['EENG 209']      
print(EECoursesQuarters)

# It is easy to iterate over the keys in a dictionary
for course, creditnumbers in EECoursesCredits.items():
    print('EE course {} has {} credits'.format(course, creditnumbers))
print('------------')

# Dictionary comprehensions: These are similar to list comprehensions, 
# but allow you to easily construct dictionaries. For example:
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)
print('------------')

#---------- 3.3 sets ----
# A set is an unordered collection of distinct elements. As a simple example, 
# consider the following:
EECourseNumbers = {'EENG 209', 'EENG 210', 'EENG 320', 'EENG 321'}
print('EENG 210' in EECourseNumbers)   
print('EENG 350' in EECourseNumbers)   
print('------------')
  
EECourseNumbers.add('EENG 350')     
print('EENG 350' in EECourseNumbers)   
print(len(EECourseNumbers))      
print('------------')
      
EECourseNumbers.remove('EENG 320')    # Remove an element from a set
print(EECourseNumbers, len(EECourseNumbers))      
print('------------')

# Loops: Iterating over a set has the same syntax as iterating over a list; 
# however since sets are unordered, you cannot make assumptions about the 
# order in which you visit the elements of the set.
EECourseNumbers = {'EENG 209', 'EENG 210', 'EENG 320', 'EENG 321', 'EENG 160'}
for idx, coursenumber in enumerate(EECourseNumbers):
    print('#{}: {}'.format(idx + 1, coursenumber))
print('------------')

# Set comprehensions: Like lists and dictionaries, we can easily construct sets 
# using set comprehensions:
from math import sqrt
print({int(sqrt(x)) for x in range(30)})
print('------------')

#----------- 3.4 Tuples ----
# A tuple is an (immutable, i.e., can't be changed after it is created)) ordered 
# list of values. Make sure that a tuple is created with parenthesis (i.e., ()). 
# A tuple is in many ways similar to a list; one of the most important differences 
# is that tuples can be used as keys in dictionaries and as elements of sets, 
# while lists cannot.
tuple1 = (0, 1, 2, 3) 
# tuple1[0] = 4 # It makes an error because "tuple" is immutable.
print(tuple1)
# Notice that types of list", "dict", "set" are mutable objects but
# built-in types such as "int", "float", "bool", "string", "unicode", "tuple" 
# are immutable objects.
print('------------')
   
d = {(x, x + 1): x for x in range(10)}  # Create a dictionary with tuple keys
t = (5, 6)       # Create a tuple
print(d)
print(type(t))
print(d[t])       
print(d[(1, 2)])
#t[0] = 1 # It makes an error because "tuple" is immutable.
print('------------')

#-------------- 4. Making Python Functions
# Python functions are defined using the "def" keyword and indentation as below:
def EvenOrOdd(x):
    if (x % 2) == 0:
        return 'Even number!'
    elif (x % 2) == 1: 
        return 'Odd number!'
    else:
        return 'Not a integer nubmer!'

for x in [3, 10, 1.3, 3.5, 7]:
    print(EvenOrOdd(x))
print('------------')
    
# Python functions could take optional keyword arguments, like this:
def EWUCampus(ProgramName, Spokane=False):
    if Spokane:
        print(f'In EWU, {ProgramName} program is located at Spokane campus')
    else:
        print('In EWU, {} program is located at Cheney campus'.format(ProgramName))

EWUCampus('ME')
EWUCampus('EE', Spokane=True)
print('------------')

#-------------- 5. Making Python Classes
# Python class can be defined using "class" keyword syntax as below:
class QuotientAndRemainder:

    # Constructor
    def __init__(self, Quotient=True, Remainder=False):
        self.Q = Quotient 
        self.R = Remainder

    # Instance method
    def QandR_Finder(self, Num, Den):
        if (self.Q and self.R):
            print('Quotient = {} and Remainder = {}'.format(Num // Den, Num % Den))
        elif (self.Q and (not self.R)):
            print('Quotient = {} '.format(Num // Den))
        elif ((not self.Q) & self.R):
            print('Remainder = {}'.format(Num % Den))
        else:
            print('There is no anything to answer !!')

QandR1 = QuotientAndRemainder(False, Remainder=True)  # Construct an instance of the "QuotientAndRemainder" class
QandR1.QandR_Finder(121,10)   # Call an instance method; 
QandR2 = QuotientAndRemainder(Quotient=True, Remainder=True)  # Construct an instance of the "QuotientAndRemainder" class
QandR2.QandR_Finder(121,10)   # Call an instance method;
print('------------ End of the basics for Python ---------')

# ---------------------- Basics for Numpy -------------------------------------
# Numpy is very useful library similar to MATLAB. If you are used to MATLAB,
# please use this tutorial at https://numpy.org/doc/stable/user/numpy-for-matlab-users.html

# To use Numpy, we first need to import the numpy package:
import numpy as np

#-------------- 1.1 Array creating
# We can initialize numpy arrays from nested Python lists, and access elements
# using square brackets with index number starting from "0":
a = np.array([[ 1., 2., 3.],[ 4., 5., 6.]])
print(f'dim = {a.ndim}, shpae = {a.shape}, size = {a.size}, type={a.dtype} \n')
print('------------')

c = np.array( [ [1,2], [3,4] ], dtype=complex )
print('------------')

# Numpy also provides many functions to create arrays:
a = np.zeros((2,2))  
print(a)
b = np.ones((1,2))   
print(b)
c = np.full((2,2), 7) 
print(c)
d = np.eye(2)        
print(d)
e = np.random.random((2,2)) 
print(e)
print('------------')
f = np.arange(12).reshape(4,3)
print(f)
print('------------')

#-------------- 1.2 Array indexing
# Similar to Numpy, you can access each element in numpy arrays with "slicing"
# of arrays.
import numpy as np
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
b = a[:2, 1:3]
print(b)
print('------------')

print(a[0, 1])
b[0, 0] = 77    
print(a[0, 1]) 
print('------------')

print(a[1,...])
print(a[...,0])
print('------------')

row1 = a[1, :]    
row2 = a[1:2, :]  
row3 = a[[1], :]  
print(row1, row1.shape, row1.ndim)
print(row2, row2.shape, row2.ndim)
print(row3, row3.shape, row3.ndim)
print('------------')

col1 = a[:,-1]
col2 = a[:,0:-1]
print(col1, col1.shape, col1.ndim)
print(col2, col2.shape, col2.ndim)
print('------------')

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
bool_idx = (a > 2)
print(bool_idx)
print(a[bool_idx])
print(a[a > 2])

#-------------- 1.3 Array arithmetic operations
# Numpy arrays works with basic arithmetic operations. It works as an element
# by element operation. If you are familiar with MATLAB, notice that "*" and 
# "/" for the numpy arrays works similarly to ".*" and "./" in MATLAB. 
x = np.reshape(np.array(range(1,5,1), dtype=np.float64),(2,2))
y = np.reshape(np.array(range(5,9,1), dtype=np.float64),(2,2))
print(x)
print(y)

print(x + y)
print(np.add(x, y))

print(x - y)
print(np.subtract(x, y))

print(x * y)
print(np.multiply(x, y))

print(x / y)
print(np.divide(x, y))

print(np.sqrt(x))
print('------------')

v = np.array([9,10])
w = np.array([11, 12])

print(v.dot(w)) 
print(np.dot(v, w))

print(v @ w)

print(x.dot(v))
print(np.dot(x, v))
print(x @ v, (x@v).shape) 

print(x.dot(y)) 
print(np.dot(x, y))
print(x @ y)
print('------------')

#-------------- 1.4. Useful methods handling arrays

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])
print(np.sum(x))  
print(np.sum(x, axis=0))  
print(np.sum(x, axis=1)) 
print('------------')

print(x)
print("transpose\n", x.T)
v = np.array([[1,2,3]])
print(v )
print("transpose\n", v.T)
print('------------')

print(np.vstack((x,y)))
print(np.hstack((x,y)))
print('------------')

print(np.concatenate((x,y),axis=0))
print(np.concatenate((x,y),axis=1))
print('------------')

print(np.tile(v,[1,3]))
print(np.tile(v,[3,1]))
print('------------')

a = np.arange(30)
b = a.reshape((2, -1, 3))  # -1 means "whatever is needed"
print(b)
print('------------')

#-------------- 2. Substitution and deep copy
# Different array objects can share the same data. Simple substitution shares
# same data. Hence, the "copy" method will NOT share the data. The "copy" 
# method makes a complete copy of the array and its data. Please understand 
# this because many bugs/errors in handling arrays in numpy are from this.
print(f'x = {x}')
xx = x
xcopy = x.copy()
x += 1
print(f'x = {x}')
print(f'xx = {xx}') # Make sure "xx" is changed by the change of "x"
print(f'xcopy = {xcopy}') # But, "xcopy" is NOT changed by the change of "x"
print(xx is x)
print(xcopy is x)
print('------------')

#-------------- 3. Broadcasting
# Broadcasting is a very useful way in array arithmetic opeation but you have 
# to be very careful in using this. If not, many bug ond/or errors are from 
# this parts. If you understand "broadcasting" well, many arithmetic operations
# with numpy arrays can be simplified as shown below examples:

x = np.reshape(np.arange(15),(5,-1))
y = np.array([10,0,10])
z = np.empty_like(x)   

# Add the vector y to each row of the matrix x with an explicit loop
for i in range(x.shape[0]):
    z[i, :] = x[i, :] + y
print(z)
print('------------')

# Above iteration loop can be simplified as below to save iteration time:
yy = np.tile(y, (x.shape[0], 1))  
print(f'yy = {yy}')                
z = x + yy  
print(f'z = {z}')
print('------------')

# Now, let's do the same calculation using the Numpy "broadcasting" as below:
z = x + y  # Add y to each row of x using broadcasting
print(z)
print('------------')
# For more detail explanation for Numpy "broadcasting", please see the documentation 
# at https://numpy.org/doc/stable/user/basics.broadcasting.html 
# or this explanation at https://scipy.github.io/old-wiki/pages/EricsBroadcastingDoc

# A universal function (or ufunc for short) is a function that operates on 
# ndarrays (i.e., numpy arrays) in an element-by-element fashion, supporting 
# array broadcasting, type casting, and several other standard features. 
# You can find the list of all universal functions in the documentation at 
# https://numpy.org/doc/stable/reference/ufuncs.html#available-ufuncs  

# ---------------------- Basics for Matplotlib --------------------------------
# Library called, "Matplotlib" provides many useful ways for plotting. This 
# section gives a brief introduction to the matplotlib.pyplot module.

# To use "matplotlib", import it first.
import matplotlib.pyplot as plt

t = np.arange(0.0, 5.0, 0.01)
c = np.cos(2*np.pi*t)
s = np.sin(2*np.pi*t)
# Plot the points using matplotlib
plt.close()
plt.figure(1)
plt.subplot(311); plt.plot(t, c, 'r-')
plt.title('Sine curve')
plt.subplot(312); plt.plot(t, s, 'b--')
plt.title('Cosine curve')
plt.xlabel('time [sec]')
plt.subplot(313); plt.plot(t,c,'r-',linewidth=4); plt.grid(True)
plt.subplot(313); plt.plot(t,s,'b-',linewidth=4); plt.grid(True)
plt.legend(['Sine', 'Cosine'])
# You can read much more about the "matplotlib" functions in the documentation 
# at https://matplotlib.org/stable/