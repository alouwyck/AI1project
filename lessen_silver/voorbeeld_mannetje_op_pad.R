P.a = matrix(c(0.8, 0.1, 0.1, 0, 0.5, 0.5), nrow=2, byrow=T, 
             dimnames=list(rownames=c("vooruit", "blijven"), 
                           colnames=c("pad", "water", "muur")))
print(P.a)

R = c(1, -1, -0.1)
names(R) = c("pad", "water", "muur")
print(R)

policy = c(0.95, 0.05)
names(policy) = c("vooruit", "blijven")
print(policy)

P = colSums(policy * P.a)
print(P)

R.huidig = sum(policy * rowSums(R * P.a))
print(R.huidig)