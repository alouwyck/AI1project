# states
n = 100
S = 0:n

# probability
p = 0.4

# rewards
R = c(rep(0, n), 1)

# number of iterations
niter = 1000

# initial values
V = matrix(0, nrow=n+1, ncol=niter+1) 

# value iteration 
for (i in 1:niter) { 
  
  # loop through all non-terminal states
  for (s in S[2:n]) {
    
    # possible actions
    A = 1:min(s, n-s)
    
    # possible next states
    iwin = s + A + 1
    ilose = s - A + 1
    
    # apply Bellman optimality equation
    V[s+1, i+1] = max(p * (R[iwin] + V[iwin, i]) + 
                      (1 - p) * (R[ilose] + V[ilose, i]))
  }
}

# optimal value of state 100 is 1
V[n+1, ] = 1

# plot optimal state-value function
matplot(S, V[, c(2:4, niter+1)], type="l", lty=1, col=1:4,
        xlab="S", ylab="V")
legend("topleft", legend=c(2:4, niter+1)-1, lty=1, col=1:4)

# get optimal policy
policy = c()
eps = 1e-10
for (s in S[2:n]) {
  A = 1:min(s, n-s)
  best.q = -1
  best.a = A[1]
  for (a in A) {
    iwin = s + a + 1
    ilose = s - a + 1
    q = p * (R[iwin] + V[iwin, niter+1]) + 
        (1 - p) * (R[ilose] + V[ilose, niter+1])
    if (q > (best.q + eps)) {
      best.q = q
      best.a = a
    } 
  }
  policy[s] = best.a
}

# plot optimal policy
plot(S[2:n], policy, type="s", xlab="S", ylab="optimal policy")



# difference between synchronous and in-place DP


# synchronous DP

# initial values
V = rep(0, n+1)
V.old = V
eps = 1e-6
dV = Inf
i = 0

# synchronous value iteration 
while (dV > eps) { 
  
  # loop through all non-terminal states
  for (s in S[2:n]) {
    
    # possible actions
    A = 1:min(s, n-s)
    
    # possible next states
    iwin = s + A + 1
    ilose = s - A + 1
    
    # apply Bellman optimality equation
    V[s+1] = max(p * (R[iwin] + V.old[iwin]) + 
                 (1 - p) * (R[ilose] + V.old[ilose]))
  }
  
  # update i, dV and V.old
  i = i + 1
  dV = max(abs(V-V.old))
  V.old = V

}

# print number of iterations
print(i)

# optimal value of state 100 is 1
V[n+1] = 1

# plot
plot(S, V)


# in-place DP

# initial values
V = rep(0, n+1)
eps = 1e-6
dV = Inf
i = 0

# in-place value iteration 
while (dV > eps) { 
  
  # loop through all non-terminal states
  dV = -Inf
  for (s in S[2:n]) {
    
    # possible actions
    A = 1:min(s, n-s)
    
    # possible next states
    iwin = s + A + 1
    ilose = s - A + 1
    
    # apply Bellman optimality equation
    # and calculate absolute difference
    tmp = V[s+1]
    V[s+1] = max(p * (R[iwin] + V[iwin]) + 
                 (1 - p) * (R[ilose] + V[ilose]))
    dV = max(c(dV, abs(V[s+1]-tmp)))
  }
  
  # update i
  i = i + 1

}

# print number of iterations
print(i)

# optimal value of state 100 is 1
V[n+1] = 1

# plot
lines(S, V, col="red")
legend("topleft", legend=c("synchr","in-place"), 
       lty=1, col=c("black","red"))

