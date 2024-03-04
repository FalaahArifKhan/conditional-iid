
MODELS_CONFIG = {
        'DTC':
        {
            "max_depth": [2, 5, 10, 20, 30, None],
            "min_samples_split" : [0.01, 0.02, 0.05, 0.1],
            "max_features": [0.6, 'sqrt'],
            "criterion": ["gini", "entropy"]
        },

        'RF' :
        {
            "max_depth": [3, 4, 6, 10],
            "min_samples_leaf": [1, 2, 4],
            "n_estimators": [50, 100, 500],
            "max_features": [0.6, 'sqrt']
        },

        'MLP' :
        {
             'hidden_layer_sizes':[(100,), (100,100,), (100,50,100,)],
             'activation': ['logistic', 'tanh', 'relu'],
             'solver': ['lbfgs', 'sgd', 'adam'],
             'learning_rate': ['constant', 'invscaling', 'adaptive']
         }
     }