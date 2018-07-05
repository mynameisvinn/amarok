import numpy as np

class Association_Mining(object):
    def __init__(self):
        pass
    
    def fit(self, transactions):
        self.transactions_ = transactions
        
    def predict(self, term):
        # fetch relevant associations
        unique_y = self._subset_y_given_X(term)

        # calculate lift = p(y|x) / p(y)
        D_x = self._subset_D(term)
        for y in unique_y:
            if y == term:
                pass
            else:
                print(y, self._calc_prior(y, D_x) / self._calc_prior(y, self.transactions_))

    def _subset_y_given_X(self, term):
        """
        within all possible *transactions* containing 
        *term*, find unique y's.
        """
        subsetted_transactions = self._subset_D(term)  # sublist of transaactions containing term

        # flatten into a single list and take the set
        flat_list = [item for sublist in subsetted_transactions for item in sublist]
        return list(set(flat_list))

    def _subset_D(self, term):
        """
        given a list of *transactions*, return a list
        of transactions containing *term*.
        """
        return list(filter(lambda transaction: term in transaction, self.transactions_))

    def _calc_prior(self, term, transactions):
        """
        calculate p(term).
        """
        n = np.sum([term in transaction for transaction in transactions])
        d = len(transactions)
        return n/d