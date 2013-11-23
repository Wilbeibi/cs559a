import numpy as np
from utils import tile_raster_images
import PIL.Image
import cPickle, gzip
from subprocess import call
import random
from copy import copy
import math
import sys

img_shape = (28,28) 
pix_per_img = img_shape[0]*img_shape[1] 

patch_shape = img_shape #(4,4) 
pix_per_patch = patch_shape[0]*patch_shape[1] 

set_shape = (img_shape[0]/patch_shape[0],img_shape[1]/patch_shape[1]) 

class Node(object):
  def __init__(self,parent=None,data=None,left=None,right=None):
    self.data = data
    self.left = left
    self.right = right
    self.parent = parent

  def is_full(self):
    return (self.left != None) and (self.right != None)

  def add_child(self,child_data):
    child = Node(self,child_data)
    if not self.left:
      self.left = child
    elif not self.right:
      self.right = child
    else:
      raise Exception("No room for child.")

class BinDistTree(object):
  def __init__(self,dist_func):
    self.D = dist_func
    self.root = Node()
    pass

  def walk(self,sample,k=4,insert=True):
    """ insert sample into search tree,
      returning the k nearest neighbors
      from the insertion path """

    neighboors = []

    "start at root"
    v = self.root 
    while v.is_full():
      "choose nearest child"
      D_l = self.D(sample,v.left.data)
      D_r = self.D(sample,v.right.data)
      if D_l < D_r:
        v = v.left
      else:
        v = v.right
      neighboors.append((v.data,min(D_l,D_r)))
    if insert:
      "insert"
      v.add_child(sample)

    return neighboors #[-k]

def knn(k,S):
  s = sorted(S, key=lambda n: n[1])
  rv = []
  for i in range(k):
    rv.append(s[i][0])
  return rv


def mnist_dist(s1,s2):
  sum_sq_err = np.sum(np.power(s1[0] - s2[0],2))
  return sum_sq_err

if __name__=="__main__":

  "load some data"
  f = gzip.open("/home/cmerck/proj/deepl/DeepLearningTutorials/data/mnist.pkl.gz",'rb')
  train_set, valid_set, test_set = cPickle.load(f)
  f.close()

'''
  bdt = BinDistTree(mnist_dist)

  print "Training...",
  n_train = 100
  for i in range(n_train):
    s = [train_set[0][i], train_set[1][i]] # [img, lbl]
    bdt.walk(s)
    if i % 1000 == 0:
      print "%d%%" % (i*100/n_train),
      sys.stdout.flush()
  print "DONE"

  print "Testing..."
  n_test = 200 #len(test_set[0])
  n_correct = 0
  n_correct_global = 0
  for i in range(n_test):
    s = (test_set[0][i], test_set[1][i])
    path = bdt.walk(s,insert=False)
    neighs = knn(1,path)
    c = neighs[0][1] # use nearest neighbor discovered
    if s[1] == c:
      n_correct += 1
      n_correct_global += 1 # assume global is correct
    else:
      # find exact match
      dmin = 10000
      for j in range(n_train):
        sj = [train_set[0][j], train_set[1][j]] # [img, lbl]
        dj = mnist_dist(s,sj)
        if dj < dmin:
          dmin = dj
          c_global = sj[1]
      if c_global == s[1]:
        n_correct_global += 1
  print "Accuracy: %1.02f" % (n_correct*100./n_test)
  print "Global Accuracy: %1.02f" % (n_correct_global*100./n_test)

  """
  print "Inserting %d:" % s[1]
  ...
  for nb in nbs:
    print " %d (%d)," % (nb[0][1], nb[1]),
  print
  """
  """image = PIL.Image.fromarray(tile_raster_images,
      X=np.array(train_set),
      img_shape=patch_shape, tile_shape=(1,len(sav2)),
      tile_spacing=(1, 1))
  image.save('mnist_data.png')
  call(['eog','mnist_data.png'])"""
'''
