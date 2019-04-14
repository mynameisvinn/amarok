import numpy as np

class Amarok(object):
    def __init__(self):
        pass
    
    def fit(self, transactions):
        self.transactions_ = transactions
        
    def predict(self, term):
        """
        calculate lift, defined as p(y|x) / p(y), for each co-occuring item
        """
        
        # identify items that co-occur with term
        unique_y = self._subset_y_given_X(term)

        # filter list - we're only interested in transactions containing term
        D_x = self._subset_D(term)

        # finally, compute lift for each co-occuring item
        for y in unique_y:
            if y == term:
                pass
            else:
                print(y, self._calc_prior(y, D_x) / self._calc_prior(y, self.transactions_))

    def _subset_y_given_X(self, term):
        """
        given *transactions* containing *term*, find unique y's.
        """
        subsetted_transactions = self._subset_D(term)  # a filtered list of transaactions containing term

        # flatten into a single list and take the set
        flat_list = [item for sublist in subsetted_transactions for item in sublist]
        return list(set(flat_list))

    def _subset_D(self, term):
        """
        given a list of transactions, return a filtered list of transactions containing term.
        """
        return list(filter(lambda transaction: term in transaction, self.transactions_))

    def _calc_prior(self, term, transactions):
        """
        calculate p(term), which is the probability of seeing p(term) in the
        entire sample.
        """
        n = np.sum([term in transaction for transaction in transactions])
        d = len(transactions)
        return n/d