
from utils import tile_raster_images
import PIL.Image
import cPickle, gzip, numpy as np
from subprocess import call
import random
from copy import copy
import math

img_shape = (28,28)
pix_per_img = img_shape[0]*img_shape[1]

patch_shape = img_shape #(4,4)
pix_per_patch = patch_shape[0]*patch_shape[1]

set_shape = (img_shape[0]/patch_shape[0],img_shape[1]/patch_shape[1])

def patchify_img(img):
  rv = []
  for px in range(set_shape[0]):
    for py in range(set_shape[1]):
      patch = []
      for ppx in range(patch_shape[0]):
        for ppy in range(patch_shape[1]):
          ix = patch_shape[0]*px+ppx
          iy = patch_shape[1]*py+ppy
          ii = ix + img_shape[0]*iy
          patch.append(img[ii])
      rv += [patch]
  return rv

# Load the dataset
f = gzip.open('../data/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

_samp_store = []
_train_i = 0
def get_next_samp():
  global _samp_store
  global _train_i
  if len(_samp_store) > 0:
    return np.array(_samp_store.pop(0))
  else:
    _samp_store = patchify_img(train_set[0][_train_i])
    _train_i += 1
    return np.array(_samp_store.pop(0))

# randomize basis vectors
def norm(v):
  return v/np.sqrt(v.dot(v))

N = 10
basis = []
basis_names = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

for i in range(N):
  new_basis = np.array([random.random() for p in range(pix_per_patch)])
  basis.append(norm(new_basis))
basis = np.array(basis)

basis_pop = [0 for b in basis] # popularity

def sparse_encode(x,basis,alpha=0,basis_p=None):
  ''' 
  Return sparse encoding of x using basis.
  Tweaked version of basis (with strength alpha) returned as basis_p.
  """
    

# apply update procedure
alpha = 0.01
sets = 20
train_sets = 10
set_size = 150
sav = []
for s in range(0,sets):
  sav += list(copy(basis))
  for xi in range(0,set_size):
    x = get_next_samp()
    basis_p = copy(basis)
    sav2 = []
    xp = copy(x)
    xprog = 0*x # blank
    exclude = []
    report = ''
    for k in range(3):
      A = basis.dot(xp) # compute coefficients
      for i in exclude:
        A[i] = 0 # don't consider components already visited
      
      #encorage use of less popular bases
      if random.random() > 0.9:
        for i in range(N):
          A[i] *= 1./(1.+math.log(1.+basis_pop[i]))
      
      i = np.argmax(A) # find principal basis vector index
      if A[i] == 0:
        break
      exclude += [i]
      report += basis_names[i] + str(A[i]) + ' '
      basis_pop[i] += 1
      c = A[i]*basis[i] # compute principal component
      basis_p[i] = norm(basis[i] + alpha*xp) # influence basis prime
      sav2 += [list(copy(xp))]
      sav2 += [list(copy(basis[i]))]
      #xprog += c
      #sav2 += [list(copy(xprog))]
      xp = xp - c # compute residual
    basis = basis_p # update basis
    #print report

    if 0:#xp.sum() > 0 and s==sets-1:
      # display progress
      image = PIL.Image.fromarray(tile_raster_images(
        X=np.array(sav2),
        img_shape=patch_shape, tile_shape=(1,len(sav2)),
        tile_spacing=(1, 1)))
      image.save('mnist_data.png')
      call(['eog','mnist_data.png'])

  print "Finished batch %d/%d. Qavg=" % (s+1,sets)

print basis_pop

sav = np.array(sav)

# display basis vectors
image = PIL.Image.fromarray(tile_raster_images(
  X=sav,
  img_shape=patch_shape, tile_shape=(sets,N),
  tile_spacing=(1, 1)))
image.save('mnist_data.png')
call(['eog','mnist_data.png'])




# now use the learned basis to perform unsupervised clustering

get_next_sample




