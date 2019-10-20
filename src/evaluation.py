# parallelization through multiprocessing
import pathos.multiprocessing as multiprocessing  # pickle support for lambda functions etc
import collections

# data split
from sklearn.model_selection import RepeatedKFold

# score
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

def _run_eval(X_train, X_test, y_train, y_test, condition, name, clf, fold=None, repeat=None):
    """Function for evaluating a classifier on a particular condition and
    returning the results as a dictionary.
    """
    # fit to data
    clf.fit(X_train, y_train)

    # calculate predictions
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:,1] # get probabilities for class 1 (True)

    # calculate scores
    # for binary classification auroc the choice of average method is irrelevant
    auroc = roc_auc_score(y_test, y_prob)
    # set binary for average method as binary classification task with positive class label '1'
    prec, recall, f, support = precision_recall_fscore_support(y_test, y_pred, average='binary')
    acc = accuracy_score(y_test, y_pred)

    # save to dictionary
    result = collections.OrderedDict()
    result['name'] = name
    result['condition'] = condition
    result['AUROC'] = auroc
    result['precision'] = prec
    result['recall'] = recall
    result['f1-score'] = f
    result['support'] = support  # TODO: add support for when only one class, i.e. binary
    result['accuracy'] = acc
    if fold != None:
        result['fold'] = fold
    if repeat != None:
        result['repeat'] = repeat

    # print results
    print('name: {name}, condition: {condition}, fold: {fold}, repeat: {repeat}\nresult: {result}').format(
        name=name, condition=condition, fold=fold, repeat=repeat, result=result)

    return result


def run_evaluation(X, y, groups, conditions=[], classifiers=[], n_splits=5, n_repeats=10, n_workers=1):
    """Function for running a CV evaluation over the classifiers in classifiers
    with n_splits and n_repeats (with respect to the groups).
    """
    # multiprocessing pool with n processes
    pool = multiprocessing.Pool(n_workers)

    # save results as a list of dicts (each dict per classifier*fold*repeat*condition)
    results = []

    # cross validation folds # IDEA: add stratified repeated k fold?
    r_kfold = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)

    # iterate over all k-folds
    for i, (train_index, test_index) in enumerate(r_kfold.split(X, y, groups)):
        # get train and test data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        fold = (i % n_splits) + 1
        repeat = int(i / n_splits) + 1

        # iterate over all conditions
        for j, condition in enumerate(conditions):
            # select condition of y
            y_train_condition = y_train[:, j]
            y_test_condition = y_test[:, j]

            # iterate over all classifiers
            for k in classifiers:  # classifiers is a dict
                clf = classifiers[k]
                # run evaluation for specific fold*repeat*condition*classifier
                result = pool.apply_async(func=_run_eval,
                                          args=(X_train, X_test, y_train_condition, y_test_condition),
                                          kwds=dict(condition=condition, name=k, clf=clf, fold=fold, repeat=repeat)
                                          )
                results.append(result)

    # get results from async call and save in results
    results = [r.get() for r in results]
    return results
