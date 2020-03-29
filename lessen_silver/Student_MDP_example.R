# STUDENT MDP ####


## STATE VALUE FUNCTION ####

# states
states = c("C1", "C2", "C3", "FB", "S")
nstates = length(states)

# actions
actions = c("S", "Z", "F", "Q", "P")
nactions = length(actions)

# policy
policy = matrix(0, nrow=nstates, ncol=nactions, 
                dimnames=list(rownames=states,
                              colnames=actions))
policy["C1", "S"] = 0.5
policy["C1", "F"] = 0.5
policy["C2", "S"] = 0.5
policy["C2", "Z"] = 0.5
policy["C3", "S"] = 0.5
policy["C3", "P"] = 0.5
policy["FB", "F"] = 0.5
policy["FB", "Q"] = 0.5
print(policy)

# state transition matrix
P.a = array(0, dim=c(nstates, nactions, nstates),
            dimnames=list(from=states,
                          action=actions,
                          to=states))
P.a["C1", "S", "C2"] = 1
P.a["C1", "F", "FB"] = 1
P.a["C2", "S", "C3"] = 1
P.a["C2", "Z", "S"] = 1
P.a["C3", "S", "S"] = 1
P.a["C3", "P", "C1"] = 0.2
P.a["C3", "P", "C2"] = 0.4
P.a["C3", "P", "C3"] = 0.4
P.a["FB", "F", "FB"] = 1
P.a["FB", "Q", "C1"] = 1
print(P.a)

# reduced state transition matrix
P = matrix(nrow=nstates, ncol=nstates,
           dimnames=list(rownames=states,
                         colnames=states))
for (state in states) {
  P[state, ] = policy[state, ] %*% P.a[state, , ]
}
print(P)

# reward matrix
R.a = matrix(0, nrow=nstates, ncol=nactions,
             dimnames=list(rownames=states,
                           colnames=actions))
R.a["C1", "S"] = -2
R.a["C1", "F"] = -1
R.a["C2", "S"] = -2
R.a["C2", "Z"] = 0
R.a["C3", "S"] = 10
R.a["C3", "P"] = 1
R.a["FB", "F"] = -1
R.a["FB", "Q"] = 0
print(R.a)

# reduced reward vector
R = rowSums(policy * R.a)
print(R)

# identity matrix
I = diag(1, nstates, nstates)

# discount factors
gamma = seq(0, 1, length.out=11)

# state value function
v = matrix(nrow=nstates, ncol=length(gamma), 
           dimnames=list(rownames=states,
                         colnames=gamma))
for (i in 1:length(gamma)) {
  v[, i] = solve(I - gamma[i]*P, R)
}

# solutions
print(v)


## ACTION VALUE FUNCTION ####


# relevant state-actions
sanames = c("C1_S", "C1_F", "C2_S", "C2_Z", "C3_S", 
            "C3_P", "FB_F", "FB_Q")
nsa = length(sanames)
sa = strsplit(sanames, "_")
names(sa) = sanames

# reward vector
R.q = rep(0, nsa)
names(R.q) = sanames
for (rname in sanames) {
  state = sa[[rname]][1]
  action = sa[[rname]][2]
  R.q[rname] = R.a[state, action]
}
print(R.q)

# probability matrix
P.q = matrix(0, nrow=nsa, ncol=nsa,
             dimnames=list(rownames=sanames,
                           colnames=sanames))
for (rname in sanames) {
  for (cname in sanames) {
    fromstate = sa[[rname]][1]
    fromaction = sa[[rname]][2]
    tostate = sa[[cname]][1]
    toaction = sa[[cname]][2]
    P.q[rname, cname] = P.a[fromstate, fromaction, tostate] * 
                        policy[tostate, toaction]
  }
}
print(P.q)

# identity matrix
I.q = diag(1, nsa, nsa)

# action value function
q = matrix(nrow=nsa, ncol=length(gamma), 
           dimnames=list(rownames=sanames,
                         colnames=gamma))
for (i in 1:length(gamma)) {
  q[, i] = solve(I.q - gamma[i]*P.q, R.q)
}

# solutions
print(q)

