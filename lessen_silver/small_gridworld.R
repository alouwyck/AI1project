# grid dimension
n = 4

# states
ns = n^2
states = seq(0, ns-1, length.out=ns)

# actions
actions = c("n", "e", "s", "w")
na = length(actions)

# policy
policy = 0.25

# reward
r = -1

# function to convert (row, col) into state index
state = function(row, col) (row-1)*n + col

# state transition matrix
P.a = array(0, dim=c(ns, na, ns), 
            dimnames=list(from=states,
                          action=actions,
                          to=states))
for (row in 1:n) {
  for (col in 1:n) {
    from = state(row, col) # current state
    # action "north"
    if (row == 1) P.a[from, "n", from] = 1
    else P.a[from, "n", state(row-1, col)] = 1
    # action "east"
    if (col == n) P.a[from, "e", from] = 1
    else P.a[from, "e", state(row, col+1)] = 1
    # action "south"
    if (row == n) P.a[from, "s", from] = 1
    else P.a[from, "s", state(row+1, col)] = 1  
    # action "west"
    if (col == 1) P.a[from, "w", from] = 1
    else P.a[from, "w", state(row, col-1)] = 1
  }
}
P.a[1, , ] = 0
P.a[ns, , ] = 0

# reduced state transition matrix
P = matrix(nrow=ns, ncol=ns,
           dimnames=list(rownames=states,
                         colnames=states))
for (i in 1:ns) {
  P[i, ] = colSums(policy * P.a[i, , ])
}
print(P)
print(rowSums(P))

# reduced reward vector
R = rep(r, ns)
R[c(1, ns)] = 0
print(R)

# discount factor
gamma = 1

# direct solution
I = diag(1, ns, ns)
v.pi = solve(I - gamma*P, R)
v.pi = matrix(v.pi, nrow=n, byrow=T)
print(v.pi)

# iterative policy evaluation
niter = 1000
v = matrix(0, nrow=ns, ncol=niter+1, 
           dimnames=list(rownames=states,
                         colnames=0:niter))
for (i in 1:niter) {
  v[, i+1] = R + gamma * P %*% v[, i]
}
print(v[, c(1:4, 11, niter+1)])

# terminal states
is.terminal = rep(F, ns)
is.terminal[c(1, ns)] = T

# add -Inf boundaries to v.pi
v.pi = rbind(rep(-Inf, n), v.pi, rep(-Inf, n))
v.pi = cbind(rep(-Inf, n+2), v.pi, rep(-Inf, n+2))

# function to convert (row+1, col+1) into state index
state = function(row, col) (row-2)*n + col-1

# get policy
policy.star = matrix(0, nrow=ns, ncol=na, 
                     dimnames=list(rownames=states,
                                   colnames=actions))
for (row in 2:(n+1)) {
  for (col in 2:(n+1)) {
    from = state(row, col) # current state
    if (!is.terminal[from]) {
      v.pi.states = c(v.pi[row-1, col],  # north
                      v.pi[row, col+1],  # east
                      v.pi[row+1, col],  # south
                      v.pi[row, col-1])  # west
      v.pi.states = round(v.pi.states)
      v.pi.max = max(v.pi.states)
      i = v.pi.states == v.pi.max
      policy.star[from, actions[i]] = 1 / sum(i)
    }
  }
}
print(policy.star)
