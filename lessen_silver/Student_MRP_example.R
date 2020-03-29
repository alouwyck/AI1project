# states
states = c("C1", "C2", "C3", "Pass", "Pub", "FB", "Sleep")
nstates = length(states)

# reward vector R
R = matrix(c(-2, -2, -2, 10, 1, -1, 0), nrow=nstates, ncol=1, 
           dimnames=list(rownames=states))
print(R)

# identity matrix I
I = diag(1, nrow=nstates, ncol=nstates)
print(I)

# discount factors gamma
gamma = seq(0, 1, length.out=11)
print(gamma)

# state transition matrix P
P = matrix(0, nrow=nstates, ncol=nstates, 
           dimnames=list(rownames=states, colnames=states))
P["C1", "C2"] = 0.5
P["C1", "FB"] = 0.5
P["C2", "C3"] = 0.8
P["C2", "Sleep"] = 0.2
P["C3", "Pass"] = 0.6
P["C3", "Pub"] = 0.4
P["Pass", "Sleep"] = 1
P["Pub", "C1"] = 0.2
P["Pub", "C2"] = 0.4
P["Pub", "C3"] = 0.4
P["FB", "C1"] = 0.1
P["FB", "FB"] = 0.9
#P["Sleep", "Sleep"] = 1
print(P)
print(rowSums(P))

# solving the system for different gamma
v = matrix(nrow=nstates, ncol=length(gamma), 
           dimnames=list(rownames=states, colnames=gamma))
for (i in 1:length(gamma)) {
  v[, i] = solve(I - gamma[i]*P, R)
}

# solutions
print(v)

