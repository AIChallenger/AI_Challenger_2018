import tensorflow as tf


def scope_wrapper(func, *args, **kwargs):
    """
       Decorator that scopes a function with its name. Useful for the graph
       visualization of Tensorboard.
    :param func:
    :param args:
    :param kwargs:
    :return:
    """

    def scoped_func(*args, **kwargs):
        with tf.name_scope("quat_{}".format(func.__name__)):
            return func(*args, **kwargs)

    return scoped_func
