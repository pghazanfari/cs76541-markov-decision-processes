import itertools

def gridsearch(param_dict, eval_fn):
    best = None
    for val in itertools.product(*param_dict.values()):
        params = dict(zip(param_dict, val))
        result = eval_fn(**params)
        if best is None or result > best[0]:
            best = (result, params)
    return best