import logging
from collections import Counter



class BalancerRatio:
    def __init__(self,target_ratio,tol):
        self.target_ratio = target_ratio
        self.tol = tol
        self.max_iter = 10000
        self.langs = ['java', 'python', 'pharo']
        self.labels = {
            'java': ['summary', 'Ownership', 'Expand', 'usage', 'Pointer', 'deprecation', 'rational'],
            'python': ['Usage', 'Parameters', 'DevelopmentNotes', 'Expand', 'Summary'],
            'pharo': ['Keyimplementationpoints', 'Example', 'Responsibilities', 'Intent', 'Keymessages', 'Collaborators']
        }


    def _split_list_into_columns(self,row, lang):
        values_list = row['labels']  # Replace 'values' with your actual column name
        dict = {}
        for key in self.labels[lang]:
            dict[key] = values_list[self.labels[lang].index(key)]

        return dict
    
    def _get_data(self):
        dd = self.ds
        return dd.map(lambda row: self._split_list_into_columns(row, self.lang))

    def balance_labels(self):
        """
        Balance positive instances per label to reach a target ratio with a tolerance.
        
        Parameters:
        - d: dict of {label: {'positive': int, 'negative': int}}
        - target_ratio: desired positive/negative ratio
        - tol: allowed relative error (default 0.1 → 10%)
        - max_iter: maximum iterations to prevent infinite loops
        
        Returns:
        - dict with final positive and negative counts
        """
        data = self._get_data()

        d = {}
        for l in self.labels[self.lang]:
            group = Counter(data[l])
            if l not in d.keys():
                d[l] = {"positive":group[1], 
                        "negative": group[0]}
                
        labels = list(d.keys())
        added_pos = {l: 0 for l in labels}
        changed = True
        iteration = 0
        
        while changed and iteration < self.max_iter:
            changed = False
            iteration += 1
            
            for l in labels:
                P = d[l]['positive'] + added_pos[l]
                N = d[l]['negative'] + sum(added_pos[other] for other in labels if other != l)
                current_ratio = P / N
                
                # Acceptable range considering tolerance
                if current_ratio < self.target_ratio * (1 - self.tol):
                    added_pos[l] += 1
                    changed = True
        
        if iteration == self.max_iter:
            logging.error("Warning: maximum iterations reached. Result may not fully meet target ratio.")
        
        # Compute final counts
        result = {}
        resultL = []
        for l in labels:
            final_pos = d[l]['positive'] + added_pos[l]
            final_neg = d[l]['negative'] + sum(added_pos[other] for other in labels if other != l)
            result[l] = {'add': added_pos[l],'positive': final_pos, 'negative': final_neg}
            resultL.append(added_pos[l])

        logging.info(result)
        
        return resultL
    
    def __call__(self,ds,lang):
        self.ds = ds
        self.lang = lang
        print('balancer')
        return self.balance_labels()