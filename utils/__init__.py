def _get_version(default='x.x.x.dev'):
    """
    新增一個可以利用 package.version query 出 build 好後版號的語法
    :param default:
    :return:
    """
    try:
        from pkg_resources import DistributionNotFound, get_distribution
    except ImportError:
        return default
    else:
        try:
            return get_distribution(__package__).version
        except DistributionNotFound:  # Run without install
            return default
        except ValueError:  # Python 3 setup
            return default
        except TypeError:  # Python 2 setup
            return default


__version__ = _get_version()
