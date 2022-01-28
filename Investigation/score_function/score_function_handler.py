from .. import score_functions

class ScoreFunctionHandler(object):
    
    def __init__(self, config):
        self.config = config
        sfunc = config.get('func', 'score_nothing')
        if isinstance(sfunc, str):
            sfunc = getattr(score_functions,sfunc)
        self.score_function = sfunc


    def score(self, results):
        return self.score_function(results, self.config)