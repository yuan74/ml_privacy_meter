import math
import scipy


# Code snipplet taken from [Steinke, Thomas, Milad Nasr, and Matthew Jagielski. "Privacy auditing with one (1) training run." Advances in Neural Information Processing Systems 36 (2024).]

# m = number of examples, each included independently with probability 0.5
# r = number of guesses (i.e. excluding abstentions)
# v = number of correct guesses by auditor
# eps,delta = DP guarantee of null hypothesis
# output: p-value = probability of >=v correct guesses under null hypothesis
def p_value_DP_audit(m, r, v, eps, delta):
  assert 0 <= v <= r <= m
  assert eps >= 0
  assert 0 <= delta <= 1
  q = 1/(1+math.exp(-eps))  # accuracy of eps-DP randomized response
  beta = scipy.stats.binom.sf(v-1, r, q)  # = P[Binomial(r, q) >= v]
  if delta == 0:
    p = beta
  else:
    alpha = 0
    sum = 0  # = P[v > Binomial(r, q) >= v - i]
    for i in range(1, v + 1):
        sum = sum + scipy.stats.binom.pmf(v - i, r, q)
        if sum > i * alpha:
          alpha = sum / i
    p = beta + alpha * delta * 2 * m
  return min(p, 1)

# m = number of examples, each included independently with probability 0.5
# r = number of guesses (i.e. excluding abstentions)
# v = number of correct guesses by auditor
# p = 1-confidence e.g. p=0.05 corresponds to 95%
# output: lower bound on eps i.e. algorithm is not (eps,delta)-DP
def get_eps_audit(m, r, v, delta, p):
  m = int(m) 
  r = int(r)
  v = int(v)
  assert 0 <= v <= r <= m
  assert 0 <= delta <= 1
  assert 0 < p < 1
  eps_min = 0  # maintain p_value_DP(eps_min) < p
  eps_max = 1  # maintain p_value_DP(eps_max) >= p
  while p_value_DP_audit(m, r, v, eps_max, delta) < p: eps_max = eps_max + 1
  for _ in range(30):  # binary search
    if eps_max - eps_min <=1e-5:
      break
    eps = (eps_min + eps_max) / 2
    if p_value_DP_audit(m, r, v, eps, delta) < p:
      eps_min = eps
    else:
      eps_max = eps
  return eps_min