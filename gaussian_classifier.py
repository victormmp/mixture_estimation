import logging
import warnings
from collections import Counter

import numpy as np
from scipy.special import softmax
from scipy.stats import beta, binom
from tqdm import tqdm
from typing import List, Dict, Union
from sklearn.neighbors import KNeighborsClassifier

log_format = "[%(asctime)s] [%(levelname)s] [%(funcName)s(%(lineno)d)] - %(message)s"
datefmt = '%d-%b-%y %H:%M:%S'

logging.basicConfig(level=logging.INFO, format=log_format, datefmt=datefmt)
warnings.filterwarnings('ignore')


def likelihood(x, mean, sigma):
    """
    Calculate likelihood of a sample with a gaussian distribution.
    Args:
        x: Sample.
        mean: Mean of the gaussian distribution.
        sigma: Covariance matrixof the gaussian distribution.
    Returns:
        The likehood value.
    """
    n = mean.shape[0]
    arg = (x - mean)
    term = np.matmul(np.matmul(arg.T, np.linalg.pinv(sigma)), arg)
    return 1 / (np.sqrt(((2 * np.pi) ** n) * np.linalg.det(sigma))) * np.exp(-0.5 * term)


def calc_statistics(x):
    """
    Calculate mean and covariance matrix for a distribution.
    Args:
        x: data distribution.
    Returns:
        Mean and covariance matrix for current distribution.
    """
    mean_1 = np.mean(x, axis=0)
    sigma_1 = np.cov(x.T)

    return mean_1, sigma_1


class ExpectationMaximization:

    def __init__(self, n_clusters: int, max_iter: int = 100, threshold: float = 0.01):
        """
        Class for fitting :n_clusters: in a distribution, using expectation-maximization algorithm.
        Args:
            n_clusters: Number of clusters to fit.
            max_iter: Maximum number of iterations allowed without converge.
            threshold: Error threshold to stop fitting.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.threshold = threshold

        self.log_likelihood = - np.inf
        self.labels = None
        self.cluster_stat = None

    def _get_clusters_statistics(self, x: np.ndarray):
        """
        Return a list of dictionaries with basic statistic for each cluster: mean, covariance and probability.
        Args:
            x: Sample matrix.

        Returns:
            None
        """
        cluster_stat = []
        N = x.shape[0]
        for cluster in range(self.n_clusters):
            c_samples = self.labels == cluster
            N_k = x[c_samples].shape[0]
            pi_k = N_k / N
            mean, sigma = calc_statistics(x[c_samples])
            cluster_stat.append(
                {
                    'mean': mean,
                    'sigma': sigma,
                    'pi_k': pi_k
                }
            )
        self.cluster_stat = cluster_stat

    def start_clusters(self, x: np.ndarray, statistics: List = None):
        """
        Initialize the EM algorithm with random clusters centered on random samples from the sample matrix, with an
        eye covariance matrix.
        Args:
            x: Sample matrix.
            statistics: (Optional) Warm Start EM algorithm with clusters.

        Returns:
            None.
        """

        if statistics:
            if len(statistics) != self.n_clusters:
                raise AttributeError(f"Clusters statistics list passed (cluster_stat) have {len(statistics)} "
                                     f"clusters, and the EM class expects {self.n_clusters} clusters.")

            self.cluster_stat = statistics
            return self

        self.labels = np.random.randint(self.n_clusters, size=x.shape[0])

        cluster_stat = []
        means = x[np.random.choice(x.shape[0], self.n_clusters, replace=False), :]
        for cluster in range(self.n_clusters):
            pi_k = 1 / self.n_clusters
            mean = means[cluster]
            sigma = np.eye(x.shape[1])
            cluster_stat.append(
                {
                    'mean': mean,
                    'sigma': sigma,
                    'pi_k': pi_k
                }
            )
        self.cluster_stat = cluster_stat

        return self

    def _log_likelihood(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the log-likelihood used as objective
        Args:
            x:

        Returns:

        """
        likelihoods = []
        for sample in x:
            likelihoods.append(np.sum([c['pi_k'] * likelihood(sample, c['mean'], c['sigma']) for c in self.cluster_stat]))
        return np.sum(np.log(likelihoods))

    def fit(self, x: np.ndarray, statistics: List = None):
        logging.info(f"Calculating EM with {self.n_clusters} clusters.")
        self.start_clusters(x, statistics)
        tau = []
        error = np.inf
        for step in range(self.max_iter):
            tau = []
            for data in x:
                likelihoods = np.array([c['pi_k'] * likelihood(data, c['mean'], c['sigma']) for c in self.cluster_stat])
                sum_clusters = np.sum(likelihoods)
                tau.append(likelihoods / sum_clusters)
            tau = np.array(tau)

            cluster_stat = []
            weight = np.sum(tau.flatten())
            for c in range(self.n_clusters):
                weight_c = np.sum(tau[:, c])
                tau_c = tau[:, c].reshape((x.shape[0], 1))
                mean = 1 / weight_c * np.sum(tau_c * x, axis=0)
                sigma = 1 / weight_c * sum([tau_c[sample] * np.matmul((x[sample] - mean).reshape(x.shape[1], 1), (x[sample] - mean).reshape(1, x.shape[1])) for sample in range(x.shape[0])])
                pi_k = weight_c / weight
                cluster_stat.append(
                    {
                        'mean': mean,
                        'sigma': sigma,
                        'pi_k': pi_k
                    }
                )
            self.cluster_stat = cluster_stat

            log_likelihood = self._log_likelihood(x)
            error = log_likelihood - self.log_likelihood
            self.log_likelihood = log_likelihood
            if error < self.threshold:
                break

            logging.info(f"Iteration {step + 1} - log-likehihood = {self.log_likelihood:10.3f} error = {error:.4}")

        logging.info(f"Expectation-maximization algorithm converged in {step} iterations with log-likelihood = {self.log_likelihood} "
                     f"and error = {error}")
        self.labels = np.argmax(softmax(tau, axis=1), axis=1)

        return self


class PartialExpectationMaximization:

    def __init__(self, cluster_statistics: List, max_iter: int = 100, threshold: float = 0.01):
        """
        Class for fitting :n_clusters: in a distribution, using expectation-maximization algorithm.
        Args:
            n_clusters: Number of clusters to fit.
            max_iter: Maximum number of iterations allowed without converge.
            threshold: Error threshold to stop fitting.
        """
        self.max_iter = max_iter
        self.threshold = threshold

        self.log_likelihood = - np.inf
        self.labels = None
        self.cluster_stat = cluster_statistics

    def _log_likelihood(self, x: np.ndarray, stat: List) -> Union[float, np.ndarray]:
        """
        Calculate the log-likelihood used as objective
        Args:
            x:

        Returns:

        """
        likelihoods = []
        for sample in x:
            likelihoods.append(np.sum([c['pi_k'] * likelihood(sample, c['mean'], c['sigma']) for c in stat]))
        return np.sum(np.log(likelihoods))

    def _fit_new_cluster(self, x: np.ndarray, c_stat: Dict):
        tau = []
        error = np.inf
        complete_stat = self.cluster_stat + [c_stat]
        for step in range(self.max_iter):
            logging.info(f"Iteration {step + 1} - Expectation with new cluster")
            tau = []
            for data in x:
                likelihoods = np.array([c['pi_k'] * likelihood(data, c['mean'], c['sigma']) for c in complete_stat])
                sum_clusters = np.sum(likelihoods)
                tau.append(likelihoods / sum_clusters)
            tau = np.array(tau)

            logging.info(f"Iteration {step + 1} - Maximization new cluster")

            weight = np.sum(tau.flatten())  # TODO: Pay attention if the weight like this can be used with partial data

            weight_c = np.sum(tau[:, -1])
            tau_c = tau[:, -1].reshape((x.shape[0], 1))
            mean = 1 / weight_c * np.sum(tau_c * x, axis=0)
            sigma = 1 / weight_c * sum([tau_c[sample] * np.matmul((x[sample] - mean).reshape(x.shape[1], 1), (x[sample] - mean).reshape(1, x.shape[1])) for sample in range(x.shape[0])])
            pi_k = weight_c / weight
            cluster_stat = [
                {
                    'mean': mean,
                    'sigma': sigma,
                    'pi_k': pi_k
                }
            ]

            complete_stat = self.cluster_stat + cluster_stat

            log_likelihood = self._log_likelihood(x, complete_stat)
            error = log_likelihood - self.log_likelihood
            self.log_likelihood = log_likelihood
            if error < self.threshold:
                break

        logging.info(f"Expectation-maximization algorithm converged with log-likelihood = {self.log_likelihood} "
                     f"and error = {error}")
        self.labels = np.argmax(softmax(tau, axis=1), axis=1)

        return self.log_likelihood, cluster_stat

    def get_best_cluster(self, x, new_clusters: List) -> Dict:
        log_likelihoods = []
        fitted_clusters = []
        for i, c in enumerate(new_clusters):
            logging.info(f"Fitting cluster {i +1} from {len(new_clusters)} clusters with Partial EM for {len(self.cluster_stat)} clusters.")
            fitted_cluster = self._fit_new_cluster(x, c)
            fitted_clusters += fitted_cluster[1]
            log_likelihoods.append(fitted_cluster[0])

        best = np.argmax(log_likelihoods)

        return fitted_clusters[best]


class GaussianMixture:

    def __init__(self, clusters: List = (3, 3)):
        """
        Gaussian mixture Model.
        Args:
            clusters: number of clusters to use for each class. Considering binary problems,
                it is a list (or tuple) with two int values.
        """
        self.clusters = clusters
        self.labels = None
        self.k = None
        self.statistics_c1 = None
        self.statistics_c2 = None

    @staticmethod
    def get_statistics(x: np.ndarray, k: int) -> List:
        """
        (Static) Get statistics for gaussian mixture for a distribution.
        Args:
            x: Data distribution matrix.
            k: number of clusters to split data.
        Returns:
            A list of dictionaries, for each of the k clusters, with its mean, covariance matrix and
            cluster probability.
        """
        em = ExpectationMaximization(k)
        em.fit(x)
        return em.cluster_stat

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Fit the gaussian mixture model.
        Args:
            x_train: Training samples.
            y_train: Training samples labels.
        """
        labels = list(Counter(y_train).keys())
        self.labels = {'class_1': labels[0], 'class_2': labels[1]}

        p_class_1 = x_train[y_train == labels[0], :].shape[0]
        p_class_2 = x_train[y_train == labels[1], :].shape[0]

        self.k = p_class_1 / p_class_2

        self.statistics_c1 = self.get_statistics(x_train[y_train == labels[0], :], self.clusters[0])
        self.statistics_c2 = self.get_statistics(x_train[y_train == labels[1], :], self.clusters[1])

    @staticmethod
    def clusters_likelihood(x: np.ndarray, statistics: List) -> np.ndarray:
        """
        (Static) Calculate the likelihood of a single sample with a gaussian mixture distribution.
        Args:
            x: Sample.
            statistics: List of dictionaries containing mean, the covariance matrix and the
                cluster probability for each cluster of the gaussian mixture.
        Returns:
            The gaussian mixture likelihood of the sample.
        """
        p_x_c = []
        for c in statistics:
            p_x_c.append(
                c['pi_k'] * likelihood(x, c['mean'], c['sigma'])
            )
        return np.sum(p_x_c)

    def bayes_decision_index(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the bayes index as a ratio between the likelihoods of a sample with the two classes
        of the system.
        Args:
            x: The Sample.
        Returns:
            The bayes index.
        """
        return self.clusters_likelihood(x, self.statistics_c1) / self.clusters_likelihood(x, self.statistics_c2)

    def gaussian_classifier(self, x: np.ndarray) -> object:
        """
        Classifies a sample according to its bayes index.
        Args:
            x: Sample.
        Returns:
            The predicted class of the sample.
        """
        return self.labels['class_1'] if self.bayes_decision_index(x) >= self.k else self.labels['class_2']

    def predict(self, x: np.ndarray) -> List:
        """
        Batch predict of the gaussian mixture model classifier.
        Args:
            x: Sample matrix.
        Returns:
            Predictions array for each sample.
        """
        predictions = []
        for sample in tqdm(x):
            predictions.append(self.gaussian_classifier(sample))
        return predictions

    def predict_indexes(self, x) -> List:
        """
        Return bayes prediction index for a batch on input files.
        Args:
            x: Sample matrix.

        Returns:
            Bayes prediction index array.
        """
        predictions = []
        for sample in tqdm(x):
            predictions.append(self.bayes_decision_index(sample))
        return predictions


class GreedyLearningGM:

    def __init__(self, n_components: int = 10, max_clusters: int = 20):

        # if n_components % 2 != 0:
        #     raise ValueError(f"Number of candidate components (n_components) must be even, not {n_components}.")



        self.n_components: int = n_components
        self.max_clusters: int = max_clusters
        self.cluster_stat: List = None

    def _generate_new_components(self, x, em: ExpectationMaximization) -> List:
        new_clusters = []
        for _ in range(self.n_components//2):
            for index, c in enumerate(em.cluster_stat):
                if x[em.labels == index, :].shape[0] > 0:
                    x_c = x[em.labels == index, :]
                    x_candidates = x_c[np.random.choice(x_c.shape[0], size=2, replace=False), :]
                    knn = KNeighborsClassifier(n_neighbors=1)
                    candidate_samples = knn.fit(x_candidates, [0, 1]).predict(x_c)
                    if x_c[candidate_samples == 0].shape[0] > 0:
                        mean, sigma = calc_statistics(x_c[candidate_samples == 0])
                        new_clusters.append({
                            'mean': mean,
                            'sigma': sigma,
                            'pi_k': c['pi_k'] / 2
                        })

                    if x_c[candidate_samples == 1].shape[0] > 0:
                        mean, sigma = calc_statistics(x_c[candidate_samples == 1])
                        new_clusters.append({
                            'mean': mean,
                            'sigma': sigma,
                            'pi_k': c['pi_k'] / 2
                        })

        return new_clusters

    def fit(self, x: np.ndarray):
        em = ExpectationMaximization(n_clusters=1)
        em.fit(x)
        cluster_stat = em.cluster_stat
        while len(cluster_stat) < self.max_clusters:
            logging.info(f"Fitting {len(cluster_stat)} clusters.")
            new_clusters = self._generate_new_components(x, em)
            pem = PartialExpectationMaximization(em.cluster_stat)
            best_cluster = pem.get_best_cluster(x, new_clusters)
            cluster_stat.append(best_cluster)
            em = ExpectationMaximization(n_clusters=len(cluster_stat))
            em.fit(x, cluster_stat)

        self.cluster_stat = cluster_stat


class StepGM:

    def __init__(self, max_clusters: int = 20):
        self.max_clusters = max_clusters
        self.cluster_stat = None
        self.n_clusters = None
        self.em = None

    @staticmethod
    def _mahalanobis_distance(x_c: np.ndarray, stat: Dict):
        """
        Calculate the Mahalanobis Distance for samples from a gaussian cluster.
        Args:
            x_c: Samples from cluster c.
            stat: Statistics of the cluster c.

        Returns:
            The distance vector with dimension = (num_samples in x_c).
        """
        distances = []
        for sample in x_c:
            mul_1 = np.matmul((sample - stat['mean']).reshape(1, x_c.shape[1]), stat['sigma'])
            mul_2 = np.matmul(mul_1, (sample - stat['mean']).reshape(x_c.shape[1], 1))
            distances.append(mul_2)

        return distances

    @staticmethod
    def _calculate_boundary(n: int, f: float, lmbda: float, b_type: str):
        logging.info(f"Calculating {b_type} boundry")
        results = []
        if b_type == 'lower':
            for k_1 in range(n + 1):
                bin_sum = []
                for k in range(k_1 + 1):
                    bin_sum.append(binom.pmf(k, n, f) - lmbda/2)
                bin_sum = np.sum(bin_sum)
                results.append(bin_sum)
            return np.argmin(results)

        if b_type == 'upper':
            for k_2 in range(n + 1):
                bin_sum = []
                for k in range(k_2, n+1):
                    bin_sum.append(binom.pmf(k, n, f) - lmbda/2)
                bin_sum = np.sum(bin_sum)
                results.append(bin_sum)
            return np.argmin(results)

    def _evaluate_mixture(self, x: np.ndarray, em: ExpectationMaximization):
        for index, c in enumerate(em.cluster_stat):
            x_c = x[em.labels == index, :]
            distances = self._mahalanobis_distance(x_c, c)
            distances = sorted(distances)
            f_hat = [n / x_c.shape[1] for n in range(len(distances))]
            a = c['mean'] * len(distances)
            b = (1-c['mean']) * len(distances)
            f = beta(a, b).cdf

            if len(distances) >= 100:
                lmbda = 0.99
            elif len(distances) >= 20:
                lmbda = 0.95
            else:
                lmbda = 0.90
            for index_rc, r_c in enumerate(distances):
                upper_bound = self._calculate_boundary(n=len(distances), f=f(r_c), lmbda=lmbda, b_type='upper')
                lower_bound = self._calculate_boundary(n=len(distances), f=f(r_c), lmbda=lmbda, b_type='lower')

                if (f_hat[index_rc] < (lower_bound / len(distances))) or (f_hat[index_rc] < (upper_bound / len(distances))):
                    logging.info(f"Mixture failed evaluation for {em.n_clusters} clusters")
                    return False

        return True

    def fit(self, x: np.ndarray, em_threshold: int = 0.01, em_max_iter: int = 100):

        for i in range(1, self.max_clusters):
            em = ExpectationMaximization(i, threshold=em_threshold, max_iter=em_max_iter)
            em.fit(x)

            self.n_clusters = i
            self.em = em
            self.cluster_stat = em.cluster_stat
            if self._evaluate_mixture(x, em):
                break

        logging.info(f"StepGM Algorithm converged with {em.cluster_stat} clusters.")
