## Joshua Yoerger
## Algorithms
## Term Project: QR Decomposition
## Due: 09.05.2011
###############################################################################

from functools import reduce
import math
import random

## First, some tools for manipulating vectors and matrices:
###############################################################################

def biggest(M):
    return max([max(list(map(lambda x:len(str(x)),row))) for row in M])

def printrow(row,dplace):
    fstr = '%'+str(dplace)+'s '
    print('[ '+reduce(lambda a,b:a+b,list(map(lambda x:fstr%str(x),row)))+']')
    return None

def pr1(row,dplace):
    fstr = str(dplace)+'s '
    print('[ '+reduce(lambda a,b:a+b,list(map(lambda x:str(x)%fstr,row)))+' ]')
    return None

def mtxprint(M):
    dpl=biggest(M)
    for row in M:
        printrow(row,dpl)
    return None

def sqrt(x):
    return x**(1/2)

def add_vects(v1,v2):
    return [x+y for x,y in zip(v1,v2)]

def inner_product(v1,v2):
    return sum([x*y for x,y in zip(v1,v2)])

def norm(v):
    return sqrt(sum([x*x for x in v]))

def scale(c,v):
    return [c*x for x in v]

def is_zero(v):
    for x in range(len(v)):   
        if v[x] != 0:
            return False
    return True

def normalize(v):
    if is_zero(v):
        print("Error. Cannot normalize the zero vector")
        return None
    else:
        return scale(1/norm(v),v)

def proj(u,v):
    ## projects v onto u
    return scale((inner_product(v,u)/inner_product(u,u)),u)

def grab_column(M,n):
    ## grabs and transposes a column a_n of the matrix A
    if n > len(M[0]):
        print("Error. The matrix has fewer than "+str(n)+" columns!")
        return None
    else:
        return [M[x][n-1] for x in range(len(M))]

def transpose(M):
    return [list(x) for x in zip(*M)]

def sum_vects(vects):
    ans = []
    for k in range(len(vects[0])):
        ans = ans + [sum(vects[j][k] for j in range(len(vects)))]
    return ans

def rem_row(M,row):
    copy = [r[:] for r in M]
    return [copy[i] for i in range(len(copy)) if i != row-1] 

def rem_zero_rows(M):
    for x in range(len(M)):
        if is_zero(M[x]):
            return rem_zero_rows(rem_row(M,x+1))
    return M

def nrows(M):
    return len(M)

def ncols(M):
    return len(M[0])

def randmtx(rows,cols,low_val,high_val,mtx=[]):
    ## returns a matrix of size rows x cols with pseudo-random, integer
    ## entries from the interval [low_val,high_val]. I wrote this to offset
    ## my lack of creativity when making up matrices to test functions on.
    for i in range(rows):
        mtx = mtx + [[random.randrange(low_val,high_val) for j in range(cols)]]
    return mtx

def mtx_mult(M1,M2):
    return [[inner_product(r,c) for r,c in zip([r for i in range(nrows(M2))],
                                                  transpose(M2))] for r in M1]

###############################################################################
##
## My implementation of the classical (i.e. numerically unstable) Gram-Schmidt
## algorithm. Actually appears pretty stable, though I'm not sure how to
## adequately test this. Notice that the matrix M is first transposed since 
## it is somewhat less cumbersome to work with rows than with columns. At the
## end of the process we transpose again restoring the original orientation. 
## The column space of Q is the same as that of M, but now the columns are
## linearly indenpendent unit length (normalized) vectors.
##
###############################################################################

def classical_gram_schmidt(M):
    V = transpose(M)
    Q = [V[0]]
    for j in range(1,len(V)):
        Q = Q + [add_vects(V[j],scale(-1,sum_vects([proj(Q[x],V[j]) for x in range(j)])))]
    Q = rem_zero_rows(Q)
    return transpose([normalize(Q[x]) for x in range(len(Q))])

###############################################################################
##
## The final product - a QR decomposition algorithm. It first computes Q by
## means of classical Gram-Schmidt algorithm above. Next we take advantage of 
## the fact that (Q^T)*(Q) = I to compute R. If we want M = QR notice that
## (Q^T)*M = (Q^T)*Q*R = I*R = R. Thus R = (Q^T)*M. After computing Q and R we
## compute Q*R and print all four matrices.
##
###############################################################################

def QR_decomp(M):
    if nrows(M) != ncols(M):
        print("This algorithm decomposes only square, real-valued matrices.")
        return None
    else:
        Q = classical_gram_schmidt(M)
        R = mtx_mult(transpose(Q),M)
        QR = mtx_mult(Q,R)
        print("Original matrix = ")
        mtxprint(M)
        print("Q = ")
        mtxprint(Q)
        print("R = ")
        mtxprint(R)
        print("Checking: Q*R = ")
        mtxprint(QR)
        return None

###############################################################################
##
## I realized after finishing the above that I could expand to non-square
## (m x n with m >= n) matrices by using complex numbers and a unitary, rather
## than orthogonal, Q. So here goes.
##
###############################################################################

def complex_conj(z):
    return [z[0],(-1)*z[1]]
    
def add_complex(z1,z2):
    return [x+y for x,y in zip(z1,z2)]

def mult_complex(z1,z2):
    return [z1[0]*z2[0] - z1[1]*z2[1],z1[0]*z2[1] + z1[1]*z2[0]]

def div_complex(z1,z2):
    if is_zero(z2):
        print("Division by zero is undefined.")
        return None
    else:
        return [(z1[0]*z2[0] + z1[1]*z2[1])/(z2[0]**2 + z2[1]**2),
                (z1[1]*z2[0] - z1[0]*z2[1])/(z2[0]**2 + z2[1]**2)]

def scale_complex_vects(c,z):
    return [mult_complex(c,x) for x in z]

def add_complex_vects(z1,z2):
    return [add_complex(x,y) for x,y in zip(z1,z2)]

def sum_complex(terms):
    ans = []
    for k in range(len(terms[0])):
        ans = ans + [sum(terms[j][k] for j in range(len(terms)))]
    return ans

def dot_complex_vects(v1,v2):
    return sum_complex([mult_complex(x,complex_conj(y)) for x,y in zip(v1,v2)])

def proj_complex_vects(u,v):
    ## projects v onto u
    return scale_complex_vects(div_complex(dot_complex_vects(v,u),
           dot_complex_vects(u,u)),u) 
    
def vect_conj(v):
    return [complex_conj(v[x]) for x in range(len(v))]

def mtx_conj(M,ans=[]):
    return [vect_conj(M[x]) for x in range(len(M))]

def mult_complex_mtx(M1,M2):
    return [[dot_complex_vects(r,c) for r,c in 
           zip([r for i in range(nrows(M2))], transpose(M2))] for r in M1]

def complex_sqrt(z):
    if z[1] < 0:
        return [sqrt((z[0] + sqrt(z[0]**2 + z[1]**2))/2),
               -sqrt((-z[0] + sqrt(z[0]**2 + z[1]**2))/2)]
    else:
        return [sqrt((z[0] + sqrt(z[0]**2 + z[1]**2))/2),
               sqrt((-z[0] + sqrt(z[0] + z[1]**2))/2)]
 
def complex_norm(v):
    return sqrt(dot_complex_vects(v,v))

def normalize_complex_vect(v):
    if is_zero_row(v):
        print("Error. Cannot normalize the zero vector")
        return None
    else:
        return scale(1/norm(v),v)

def is_zero_row(v):
    for x in range(len(v)):
        if not is_zero(v[x]):
            return False
    return True

def rem_zero_rows_complex(M):
    for x in range(len(M)):
        if is_zero_row(M[x]):
            return rem_zero_rows_complex(rem_row(M,x+1))
    return M

def complex_gram_schmidt(M):
    V = transpose(M)
    Q = [V[0]]
    for j in range(1,len(V)):
        Q = Q + [add_complex_vects(V[j],
            scale_complex_vects(-1,sum_complex([proj_complex_vects(Q[x],V[j]) 
            for x in range(j)])))]
    Q = rem_zero_rows_complex(Q)
    return transpose([normalize(Q[x]) for x in range(len(Q))])


