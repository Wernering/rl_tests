import logging
import math

from config.logger import LOG_NAME

logger = logging.getLogger(LOG_NAME)


class Location:
    def __init__(
        self,
        name: str,
        capacity: int,
        lease_lambda: int,
        return_lambda: int,
        epsilon: float,
    ):
        self.name = name

        # maximum number of cars in location
        self.capacity = capacity

        # Expected value of lease probability
        self.lease_k = lease_lambda

        # Expected value of returned car probability
        self.return_k = return_lambda

        # Minimum probability of happening
        self.epsilon = epsilon

        # Dictionary f probabilities of renting x cars
        self.lease_probability = self.dictionary_of_probabilities(self.lease_k)
        logger.info(
            f"Location {self.name}, lease probability: {self.lease_probability}"
        )

        self.return_probability = self.dictionary_of_probabilities(self.return_k)
        logger.info(
            f"Location {self.name}, return probability: {self.return_probability}"
        )

    @staticmethod
    def poisson(k, n):
        return math.exp(-k) * (k**n) / math.factorial(n)

    def dictionary_of_probabilities(self, k):
        """
        Calculates a dictionary of probabilities of renting x cars.
        The dictionary is as big as the probability is over self.epsilon
        """

        poisson_dictionary = dict()
        x = 0

        while True:
            prob = self.poisson(k, x)
            if all([x > k, prob < self.epsilon]):
                break
            poisson_dictionary[x] = prob
            x += 1

        # Normalize probability
        extra = (1 - sum(poisson_dictionary.values())) / len(poisson_dictionary)
        poisson_dictionary = {k: prob + extra for k, prob in poisson_dictionary.items()}

        return poisson_dictionary
