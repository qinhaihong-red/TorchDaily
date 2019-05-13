#%% [markdown]
# 
# # Sobel and Laplacian Operator
# 
# ## Theory
# 
# In case of one dimentional space, we use gradient $\dfrac{\partial f}{\partial x}$ to measure the rate of change in $x$ direction in which $f$ is a mapping like $f:R \to R $ and denoted as $y=f(x)$.
# 
# So does the same in case of two dimentional space, except we have gradients in 2 directions:
# 
# - direction x:$\dfrac{\partial f}{\partial x}$
# 
# 
# - direction y:$\dfrac{\partial f}{\partial y}$
# 
# where $f$ denoted as $z=f(x,y).$
# 
# The digit image can be viewed as the same mapping as : $z=f(x,y)$, in which $(x,y)$ represents the coordinates,and the $z$ means intensity value.But it is a discrete function, so how can we calculate the gradients of a image?
# 
# __The answer is to calculate difference to approximate gradient.__ Let's see it.
# 
# In one dimensional space :
# 
# - first order derivative can be approximated by difference as:$\dfrac{\partial f}{\partial x} = f(x+1)-f(x) $
# 
# - second order derivative can be approximated by diference as:
# $\dfrac{\partial^2 f}{\partial x^2} = f(x+1)-f(x) - \{f(x)-f(x-1)\} = f(x+1)+f(x-1)-2*f(x) $
# 
# And two dimensional space:
# 
# - first order derivative:
#     - in x direction : $\dfrac{\partial f}{\partial x} = f(x+1,y)-f(x,y) $
#     - in y direction : $\dfrac{\partial f}{\partial x} = f(x,y+1)-f(x,y) $
#     
# - second order derivative:
#     - in x direction:$\dfrac{\partial^2 f}{\partial x^2} = f(x+1,y)-f(x,y) - \{f(x,y)-f(x-1,y)\} = f(x+1,y)+f(x-1,y)-2*f(x,y)$
#     - in y direction:$\dfrac{\partial^2 f}{\partial y^2} = f(x,y+1)-f(x,y) - \{f(x,y)-f(x,y-1)\} = f(x,y+1)+f(x,y-1)-2*f(x,y)$
# 
# 
# __In digit image processing, in order to obtain the difference talked above, we can do convolution or correlation calculation with specific spatial filter__: 
# 
# - first order Sobel kernel : applied for first order derivatives, which can be described as below:
#     - x direction : $\begin{bmatrix}
# -1&0&1\\
# -2&0&2\\
# -1&0&1
# \end{bmatrix}$
# 
#     - y direction: $\begin{bmatrix}
# -1&-2&-1\\
# 0&0&0\\
# -1&2&1
# \end{bmatrix}$
# 
# - second order Sobel kernel : applied for second order derivatives：
#     - x direction : $\begin{bmatrix}
# 1&-2&1\\
# 4&-2&4\\
# 1&-2&1
# \end{bmatrix}$
# 
#     - y direction: $\begin{bmatrix}
# 1&4&1\\
# -2&-2&-2\\
# 1&4&1
# \end{bmatrix}$
# 
# __As for Laplacian operaiton,we can obtain the result either by summing the calculated second order Sobel difference of the image in 2 directions talked above, or doing correlation calculation with the image using the kernels descpribed as below ：__
# 
# - $\begin{bmatrix} 
# 0&1&0\\
# 1&-4&1\\
# 0&1&0
# \end{bmatrix}$ or $\begin{bmatrix}
# 1&1&1\\
# 1&-8&1\\
# 1&1&1
# \end{bmatrix}$
#%% [markdown]
# ## Practise
# 
# In OpenCV, there are 2 functions :Sobel() and Laplacian(), which implement Sobel and Laplacian Operator.
# 
# But we can use the methods talked above to do the same thing without these built-in functions for better understanding. 
# 
# Here is a simple example：
# 
# - First, we define a matrix $\boldsymbol{m}$ as:
# $\begin{bmatrix} 
# 1&2&5\\
# 4&5&11\\
# 7&8&9
# \end{bmatrix}$
# 
# 
# - using built-in function *cv::Sobel* to calculate the first derivative of $\boldsymbol{m}$ in x direction: *cv::Sobel($\boldsymbol{m}$, $\boldsymbol{dst}$, CV_16S, 1, 0)*; and the result is:$\begin{bmatrix} 
# 0&22&0\\
# 0&20&0\\
# 0&18&0
# \end{bmatrix}$
# 
# 
# - for comparison, we use built-in fucntion *cv::filter2D* with the kernel talked above:$\begin{bmatrix}
# -1&0&1\\
# -2&0&2\\
# -1&0&1
# \end{bmatrix}$ , to do correlation calculation to $\boldsymbol{m}$ : *cv::filter2D($\boldsymbol{m}$, $\boldsymbol{dst2}$, CV_16S, $\boldsymbol{filter\_x}$)*; and we can see the result $\boldsymbol{dst2}$ is same as the output $\boldsymbol{dst}$ of Sobel funciton.
# 
# 
# 
# __See the code for more details.__

#%%



