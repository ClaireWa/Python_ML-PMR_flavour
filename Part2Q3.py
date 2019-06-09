from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import scipy.stats as st


# Part A
def p_gauss(x, y):
    """
    :param x: value for x
    :param y: value for y
    :return: sample from a bivariate normal distribution
    """
    # assume mean is 0 and variance is 1
    return st.multivariate_normal.pdf([x, y])


# def mh(p_star, param_init, num_samples=5000, stepsize=1.0):
#     """
#     :param p_star: a function on theta that is proportional to the density of interest p*(theta);
#     :param param_init: the initial sample - a value for theta from where the Markov chain starts;
#     :param num_samples: the number S of samples to generate;
#     :param stepsize: a hyperparameter specifying the variance of the Gaussian proposal distribution q;
#     :return: return a list of samples from p(theta) proportional to p*(theta)
#     """
#
#     x, y = param_init[0], param_init[1]
#     samples = np.zeros((num_samples, 2))
#     # set first sample to be param_init values
#     samples[0] = np.array([x, y])
#     for i in range(num_samples)[1:]:
#         # draw a candidate from the proposal distribution (Gaussian)
#         x_cand, y_cand = np.array([x, y]) + st.norm(0, stepsize).rvs(2)
#         # Define the acceptance criteria. As gaussian is used as proposal distribution q(theta_initial|theta_cand) =
#         # q(theta_cand|theta_initial), therefore q functions cancel and can use following acceptance criterion
#         a = p_star(x_cand, y_cand) / p_star(x, y)
#         if all(a >= 1):
#             x, y = x_cand, y_cand
#         # When a<1, draw a random value, u, uniformly from the unit interval [0,1] to compare against
#         u = np.random.rand()
#         if all(a > u):
#             x, y = x_cand, y_cand
#         samples[i] = np.array([x, y])
#
#     return samples


# Part B
# number of samples = 5000, initialise at x=0, y=0, stepsize =1
# s1 = mh(p_gauss, param_init=[0, 0], num_samples=5000, stepsize=1)
# plt.scatter(s1[:, 0][20:], s1[:, 1][20:], c="b", label = "Rest of the samples")
# plt.scatter(s1[:, 0][0:20], s1[:, 1][0:20], c="r", label = "First 20 samples")
# plt.title("p(x,y) = N(x;0,1)N(y;0,1) represented by 5,000 samples, initialised at (0,0)")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend(loc = 'best', frameon = "True")
# plt.show()

# number of samples = 5000, initialise at x=7, y=7, stepsize =1
# s2 = mh(p_gauss, param_init=[7, 7], num_samples=5000, stepsize=1)
# plt.scatter(s2[:, 0][20:], s2[:, 1][20:], c="b", label = "Rest of the samples")
# plt.scatter(s2[:, 0][0:20], s2[:, 1][0:20], c="r", label = "First 20 samples")
# plt.title("p(x,y) = N(x;0,1)N(y;0,1) represented by 5,000 samples, initialised at (7,7)")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend(loc = 'best', frameon = "True")
# plt.show()


def mh(p_star, param_init, num_samples=5000, stepsize=1.0, W=0):
    """
    :param p_star: a function on theta that is proportional to the density of interest p*(theta);
    :param param_init: the initial sample - a value for theta from where the Markov chain starts;
    :param num_samples: the number S of samples to generate;
    :param stepsize: a hyperparameter specifying the variance of the Gaussian proposal distribution q;
    :param W: period between initialisation and starting to collect samples, "warm-up"
    :return: return a list of samples from p(theta) proportional to p*(theta)
    """

    x, y = param_init[0], param_init[1]
    samples = np.zeros((num_samples, 2))
    for i in range(num_samples + W):  # [1:]:
        # draw a candidate from the proposal distribution (Gaussian)
        x_cand, y_cand = np.array([x, y]) + st.norm(0, stepsize).rvs(2)
        # Define the acceptance criteria. As gaussian is used as proposal distribution q(theta_initial|theta_cand) =
        # q(theta_cand|theta_initial), therefore q functions cancel and can use following acceptance criterion
        a = p_star(x_cand, y_cand) / p_star(x, y)
        if a >= 1:
            x, y = x_cand, y_cand
        # When a<1, draw a random value, u, uniformly from the unit interval [0,1] to compare against
        u = np.random.rand()
        if a > u:
            x, y = x_cand, y_cand
        # only collect the samples after number of MCMC steps specified in W are taken
        if i >= W:
            samples[i-W] = np.array([x, y])

    return samples



# Part C


def p_(alpha, beta):
    """
    :param alpha: parameter of regression model which is assumed to be drawn from N(0,100)
    :param beta: parameter of regression model which is assumed to be drawn from N(0,100)
    :return: target density which is proportional to p(alpha,beta|D) and can be used in MH algorithm
    """
    # calculate the prior, p(alpha, beta) = p(alpha)*p(beta)
    # as alpha and beta are independently drawn from same normal distn.
    a = st.norm(0, 10).pdf(alpha)
    b = st.norm(0, 10).pdf(beta)
    prior = a * b
    # calculate the likelihood p(Data|alpha, beta)
    lik = []
    data = np.loadtxt("q3_poisson.txt", delimiter=' ', unpack=False)
    for i in range(len(data[1])):
        p = float(st.poisson.pmf(data[1][i], np.exp(alpha * data[0][i] + beta)))
        lik.append(p)
    lik = np.prod(lik)
    return lik * prior


s3 = mh(p_, param_init=[0, 0], num_samples=5000, stepsize=1, W=1000)
plt.scatter(s3[:, 0], s3[:, 1])
plt.title(r'p($\alpha$,$\beta$|D) represented by 5,000 samples')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
plt.show()

mean = np.mean(s3, axis =0)
corr = np.corrcoef(s3, rowvar = False)