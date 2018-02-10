import argparse


class Flags(object):
    args = {}
    parser = argparse.ArgumentParser()
    parsed = False

    @classmethod
    def add_arg(cls, *args, **kwargs):
        cls.parser.add_argument(*args, **kwargs)

    @classmethod
    def set_defaults(cls, **kwargs):
        cls.parser.set_defaults(**kwargs)

    @classmethod
    def parse(cls):
        cls.args = vars(cls.parser.parse_known_args()[0])

    @classmethod
    def __getattr__(cls, name):
        """Retrieves the 'value' attribute of the flag --name."""
        if not cls.parsed:
            cls.parse()
        if name not in cls.args:
            raise AttributeError(name)
        return cls.args[name]


CONFIG = Flags()
