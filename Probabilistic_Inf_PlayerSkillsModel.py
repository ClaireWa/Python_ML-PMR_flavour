import numpy as np
import matplotlib.pyplot as plt

#### Part f
s = [1, 2, 3, 4, 5] # possible values for skill of players (s1, s2 & s3)
p_s1 = [0.01, 0.01, 0.08, 0.2, 0.7] # pmf of s1 (given)
p_s2 = [0.02, 0.02, 0.06, 0.3, 0.6] # pmf of s2 (given)
p_s3 = [0.2, 0.2, 0.2, 0.2, 0.2] # pmf of s3 (given)

# vector for phi_1_dash as in part(e)
phi_1_dash = []
for s2 in s:
    for s3 in s:
        tot_sum = 0
        for s1 in s:
            sum = p_s1[s1-1]* ((1/10)*(s1-s2) + 0.5) * ((1/10)*(s3-s1) + 0.5)
            tot_sum += sum
        phi_1_dash.append(tot_sum)

# reshape phi_1_dash into array so rows will be s2 and columns will be s3
phi_1_dash = np.array(phi_1_dash)
phi_1_dash = np.reshape(phi_1_dash, (5,5))

# vector for phi_12_tilda defined as
# phi_12_tilda = Sum_s2{phi_2* phi_b_r2(s2 , s3) * phi_1_dash(s2, s3)}
# i.e. phi_12_tilda = Sum_s2{phi_2* phi_1_tilda(s2, s3)} as defined prev. in part (f)
phi_12_tilda = []
for s3 in s:
    tot_sum = 0
    for s2 in s:
        sum = p_s2[s2-1]* (1-((1/10)*(s2-s3) + 0.5)) * phi_1_dash[s2-1][s3-1]
        tot_sum += sum
    phi_12_tilda.append(tot_sum)

# rescale both vectors s.t. the max value of the elements is one
psi_2_scaled = np.array(phi_12_tilda)/max(phi_12_tilda)
psi_1_scaled = np.array(p_s3)/max(p_s3)

print(np.round(psi_2_scaled, 4))
# [0.0744 0.2069 0.4053 0.6697 1.]
print(psi_1_scaled)
# [1. 1. 1. 1. 1.]


#### Part f
# compute the normalization factor
Z = 0
for s3 in s:
    sum = psi_1_scaled[s3-1]* psi_2_scaled[s3-1]
    Z += sum
# compute the posterior
post_s3 = (psi_1_scaled*psi_2_scaled)/Z
print(np.round(post_s3, 4))
# [0.0316 0.0878 0.172  0.2842 0.4244]


# ((1/2)*(1 + np.tanh(s2-s3)))
# vector for phi_1_dash as in part(e)
phi_1_dash_ = []
for s2 in s:
    for s3 in s:
        tot_sum = 0
        for s1 in s:
            sum = p_s1[s1-1]* (((1/2)*(1 + np.tanh(s1-s2)))) * ((1/2)*(1 + np.tanh(s3-s1)))
            tot_sum += sum
        phi_1_dash_.append(tot_sum)

# reshape phi_1_dash into array so rows will be s2 and columns will be s3
phi_1_dash_ = np.array(phi_1_dash_)
phi_1_dash_ = np.reshape(phi_1_dash_, (5,5))

# vector for phi_12_tilda defined as
# phi_12_tilda = Sum_s2{phi_2* phi_b_r2(s2 , s3) * phi_1_dash(s2, s3)}
# i.e. phi_12_tilda = Sum_s2{phi_2* phi_1_tilda(s2, s3)} as defined prev. in part (f)
phi_12_tilda_ = []
for s3 in s:
    tot_sum = 0
    for s2 in s:
        sum = p_s2[s2-1]* (1-(((1/2)*(1 + np.tanh(s2-s3))))) * phi_1_dash[s2-1][s3-1]
        tot_sum += sum
    phi_12_tilda_.append(tot_sum)

# rescale both vectors s.t. the max value of the elements is one
psi_2_scaled_ = np.array(phi_12_tilda_)/max(phi_12_tilda_)
psi_1_scaled_ = np.array(p_s3)/max(p_s3)

print(np.round(psi_2_scaled_, 4))
# [0.0083 0.0386 0.1321 0.419  1.    ]
print(psi_1_scaled_)
# [1. 1. 1. 1. 1.]

# compute the normalization factor
Z_ = 0
for s3 in s:
    sum = psi_1_scaled[s3-1]* psi_2_scaled[s3-1]
    Z_ += sum
# compute the posterior
post_s3_ = (psi_1_scaled_*psi_2_scaled_)/Z_
print(np.round(post_s3_, 4))
# [0.0035 0.0164 0.0561 0.1778 0.4244]

# compare posterior and prior in a plot
Nsamp = len(post_s3)
p = np.linspace(0,1,Nsamp)
plt.figure()
plt.plot(p,post_s3,'b',label='posterior (g)')
#plt.plot(p,post_s3_,'r',label='posterior (h)')
plt.plot(p,p_s3,'k',label='prior')
plt.xlabel('p')
plt.ylabel('PMF')
plt.legend(loc='best')
plt.title("Comparison plot of the posterior and prior")
plt.show()

