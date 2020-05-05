class MyRepeatableIterator(object):
    """Repeatable iterator class."""

    def __init__(self, iter_fn):
        """Create a repeatable iterator.

        Args:
          iter_fn: callable with no arguments, creates an iterator
        """
        self._iter_fn = iter_fn
        self._counter = 0

    def get_counter(self):
        return self._counter

    def __iter__(self):
        self._counter += 1
        return self.called_iter_fn.__iter__()

    def __call__(self, *args, **kwargs):
        self.called_iter_fn = self._iter_fn(*args, **kwargs)
