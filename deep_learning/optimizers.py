from torch import optim


def torch_optimizer_factory(optim_name, lr, parameters):
    if optim_name.lower() == 'sgd':
        return optim.SGD(params=parameters, lr=lr)

    elif optim_name.lower() == 'rprop':
        return optim.Rprop(params=parameters, lr=lr)

    elif optim_name.lower() == 'adadelta':
        return optim.Adadelta(params=parameters, lr=lr)

    elif optim_name.lower() == 'adagrad':
        return optim.Adagrad(params=parameters, lr=lr)

    elif optim_name.lower() == 'adamw':
        return optim.AdamW(params=parameters, lr=lr)

    elif optim_name.lower() == 'adam':
        return optim.Adam(params=parameters, lr=lr)

    elif optim_name.lower() == 'adamax':
        return optim.Adamax(params=parameters, lr=lr)

    elif optim_name.lower() == 'asgd':
        return optim.ASGD(params=parameters, lr=lr)

    elif optim_name.lower() == 'rmsprop':
        return optim.RMSprop(params=parameters, lr=lr)

    elif optim_name.lower() == 'radam':
        return optim.RAdam(params=parameters, lr=lr)

    elif optim_name.lower() == 'nadam':
        return optim.NAdam(params=parameters, lr=lr)

    else:
        raise ValueError(f"{optim_name} is not among the supported optimizers")

