
norm = lambda v, p: sum([abs(x)**p for x in v])/p
euclideanNorm = lambda v: sum([abs(x)**2 for x in v])/2
l1Norm = lambda v: sum([abs(x) for x in v])
maxNorm = lambda v: max(v)

# norms basically measure the size of a vector, formally L^p
# f : V -> (x <= R^+)
# eucledian norm (L^2) is simply the euclidean distance from the origin
#   to the point identified by x

