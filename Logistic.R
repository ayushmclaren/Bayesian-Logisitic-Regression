#####################################################################################

## Numerical Assignment Question 2 ##
## Bayesian Logistic Regression    ##

## Prior for Beta ~ N5(0,100I5)    ##

## Yi ∼ Bern (pi) where pi = exp((xi)T %*% β)/ (1 + exp((xi)T %*% β)) ##

## To find Posterior Distribution of Beta and sampling using MCMC     ##

#####################################################################################
set.seed(42)

logpost <- function(y,x,beta,sig2,flag=0) #calculate posterior likelihood (log scale)
{
  n = length(y)
  like  = 0 
  
  for(i in 1:n)
  {
    like = like + (t(y[i]*x[i,]) %*% (beta)) - log(1 + exp(t(x[i,]) %*% (beta)))
  }
  
  if(flag==1) #for betastart (initialization)
  {
    return(like)
  }
  
  else{
        like = like - ((t(beta) %*% (beta))/(2*sig2))
        return(like)
  }
}

RMVNorm <- function(n,mu,sigma) #Draws Randomly from MVNorm
{
  p <- length(mu)
  decomp <- svd(sigma)     #using svd instead of cholesky
  sqrt.sig <- decomp$u %*% diag(sqrt(decomp$d), p) %*% t(decomp$v)
  
  sample <- matrix(0,nrow=n,ncol=p)
  for(i in 1:n)
  {
    Z <- rnorm(p, mean = 0, sd = 1)
    sample[i,] = mu + sqrt.sig %*% Z
  }
  return(sample)
}

##  Finding initialising value by sampling from Prior and  ##
##  Using the Beta = argmax(likelihood(Beta)) for the data ##

FindBetaStart <- function(beta,y,x) 
{
  likelihood <- numeric(length = nrow(beta))
  
  for(i in 1:nrow(beta))
  {
    likelihood[i] <- logpost(y=y,x=x,beta=as.matrix(beta[i,]),100,flag=1)
  }
  
  beta.init <- beta[which.max(likelihood),]
  return(beta.init)
}

BETAmcmc <- function(y,x,h,N,start,acc.prob) #MCMC Sampling
{
  chain = matrix(0,nrow = N,ncol = length(start))
  chain[1,] = start
  
  naccept <- 0
  
  for(t in 2:N)
  {
      prop <- as.vector(RMVNorm(n=1,mu=chain[t-1,],sigma=h))
      ratio <- logpost(y=y,x=x,beta=prop,sig2=100,flag=0) - logpost(y=y,x=x,beta=as.vector(chain[t-1,]),sig2=100,flag=0)
      if(runif(1,min=0,max=1) < exp(ratio))
      {
        naccept <- naccept + 1
        chain[t,] <- prop
      } 
      
      else
      {
        chain[t,] <- chain[t-1,]  
      }
  }
  
  acc.prob <<- naccept/N              #Global variable assignment
  print(paste("Acceptance probabaility = ", acc.prob))
  return(chain)
}


dat <- read.table("http://home.iitk.ac.in/~dootika/assets/course/Log_data/170187.txt",
                  header = F)
Ydat <- as.matrix(dat[,1])
Xdat <- as.matrix(dat[,-1])
N <- 1e5

#Sample from Prior
beta.sample <- RMVNorm(n=1e4,mu=rep(0,ncol(Xdat)),sigma = 100*diag(ncol(Xdat)))
init <- FindBetaStart(y=Ydat,x=Xdat,beta = beta.sample)

step = 0.155*diag(5) #Update step size
acc.prob = 0
chain <- BETAmcmc(y=Ydat,x=Xdat,h=step,N=N,start = init,acc.prob=acc.prob)

beta.est <- colMeans(chain) #Posterior Mean

par(mfrow=c(2,3)) # Auto Correlation Plots
acf(chain[,1],main="ACR Plot Component 1") 
acf(chain[,2],main="ACR Plot Component 2")
acf(chain[,3],main="ACR Plot Component 3")
acf(chain[,4],main="ACR Plot Component 4")
acf(chain[,5],main="ACR Plot Component 5")
graphics.off()

par(mfrow=c(2,3)) #Timed Plots
plot.ts(chain[,1],main="Component 1",ylab="Beta0")
plot.ts(chain[,2],main="Component 2",ylab="Beta1")
plot.ts(chain[,3],main="Component 3",ylab="Beta2")
plot.ts(chain[,4],main="Component 4",ylab="Beta3")
plot.ts(chain[,5],main="Component 5",ylab="Beta4")
graphics.off()


par(mfrow=c(2,3)) #Density Plots
plot(density(chain[,1]),main="Density Plot Component 1")
abline(v=mean(chain[,1]),col="violet")
plot(density(chain[,2]),main="Density Plot Component 2")
abline(v=mean(chain[,2]),col="blue")
plot(density(chain[,3]),main="Density Plot Component 3")
abline(v=mean(chain[,3]),col="green")
plot(density(chain[,4]),main="Density Plot Component 4")
abline(v=mean(chain[,4]),col="orange")
plot(density(chain[,5]),main="Density Plot Component 5")
abline(v=mean(chain[,5]),col="red")

print(acc.prob)
print(beta.est)
