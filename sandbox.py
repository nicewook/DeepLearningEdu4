
# coding: utf-8

# In[4]:
import numpy as np
x = np.array([[1,2], [3,4]])
for column in x:
    print(column)
for row in x:
    print(row)


# In[8]:

xx = x.flatten()
print(xx)




# In[10]:

xx[np.array([0,3])]


# In[11]:

xx < 3


# In[12]:

xx[xx>3]


# In[1]:

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,6,0.1)
y = np.sin(x)

plt.plot(x,y)
plt.show()


# In[3]:

y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label='sin')
plt.plot(x, y2, linestyle = '--', label='cos')
plt.xlabel('x')
plt.ylabel('y')
plt.title('sin and cos')
plt.legend()
plt.show()


# In[5]:

from matplotlib.image import imread
img = imread('lena.png')

plt.imshow(img)
plt.show()


# In[1]:

def AND(x1, x2):
    x = np.array([x1,x2])
    w = np.array([.5, .5])
    b = -.7
    
    tmp = np.sum(w*x) + b
    
    if tmp > 0:
        return 1
    else:
        return 0


# In[2]:

print(AND(0,0), AND(0,1), AND(1,0), AND(1,1))


# In[3]:

x = np.array([-1,-2,0, 1, 3])
print(x)
y = x > 0
print(y)
print(x[y])  # y가 true인 경우의 index를 가진 x[y] 출력


# In[4]:

import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return np.array(x>0, dtype=np.int)

x = np.arange(-5.0, 5., .1)
y = step_function(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()


# In[11]:

def sigmoid(x):
    return 1/(1 + exp(-x))


# In[12]:

x = np.array([-1., 1., 2.])
sigmoid(x)


# In[15]:

x= np.arange(-5., 5., 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-.1, 1.1)
plt.show()


# In[19]:

# 차원, shape등을 보자

a = np.array([1,2,3,4])
print(np.ndim(a))
print(a.shape)
print(a.shape[0])


# In[22]:

# 2차원 numpy array
b = np.array([[1,2], [3,4], [5,6]])
print(b)
print(np.ndim(b))
print(b.shape)
print(b.shape[0], b.shape[1])


# In[23]:

c = np.array([2,3])
print(b.dot(c))


# In[24]:

print(np.dot(b,c))


# In[29]:

X = np.array([1,2])
print(X.shape)
W=np.array([[1,3,5], [2,4,6]])
print(W)
print(W.shape)
print(X.dot(W))
print(np.dot(X,W))


# In[31]:

# softmax 공부
import numpy as np
a = np.array([.3, 2.9, 4.0])

exp_a = np.exp(a)
print(exp_a)


# In[32]:

sum_exp_a = np.sum(exp_a)
sum_exp_a


# In[33]:

y = exp_a/sum_exp_a
y


# In[34]:

y_sum = np.sum(y)
y_sum


# In[41]:

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    print(y)
    return y

def softmax2(a):
    print(np.exp(a)/np.sum(np.exp(a)))
    return np.exp(a)/np.sum(np.exp(a))


# In[42]:

# softmax의 overflow 예방
# 지수함수에 들어갈 x값을 작게 만들어준다

a = np.array([1010, 1000, 900])
print(a)
a = a - np.max(a)
print(a)

softmax(a)
softmax2(a)


# In[ ]:



