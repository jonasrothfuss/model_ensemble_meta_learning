import inspect

def get_all_function_arguments(function, locals):
    """

    :param function: callable function object
    :param locals: local variable scope of the function
    :return:
        args: list or arguments
        kwargs_dict: dict of kwargs

    """
    kwargs_dict = {}
    for arg in inspect.getfullargspec(function).kwonlyargs:
        if arg not in ['args', 'kwargs']:
            kwargs_dict[arg] = locals[arg]
    args = [locals[arg] for arg in inspect.getfullargspec(function).args]

    if 'args' in locals:
        args += locals['args']

    if 'kwargs' in locals:
        kwargs_dict.update(locals['kwargs'])
    return args, kwargs_dict
