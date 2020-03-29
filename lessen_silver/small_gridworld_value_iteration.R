# gridworld dimensions
n = 4

# reward
r = -1

# number of iterations
niter = 7

# initializing v
v = array(0, dim=c(n, n, niter))

# value iteration algorithm
for (iter in 1:(niter-1)) {
  for (irow in 1:n) {
    for (icol in 1:n) {
      # not the terminal state (1, 1)
      if (!(irow==1 & icol==1)) {
        q = rep(0, n)
        # north
        if (irow > 1) q[1] = r + v[irow-1, icol, iter]
        else q[1] = r + v[irow, icol, iter]
        # south
        if (irow < n) q[2] = r + v[irow+1, icol, iter]
        else q[2] = r + v[irow, icol, iter]
        # east
        if (icol < n) q[3] = r + v[irow, icol+1, iter]
        else q[3] = r + v[irow, icol, iter]
        # west
        if (icol > 1) q[4] = r + v[irow, icol-1, iter]
        else q[4] = r + v[irow, icol, iter]
        # maximum
        v[irow,icol, iter+1] = max(q) 
      }
    }
  }
  print(v[, , iter+1])
}


