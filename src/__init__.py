import os

README_path = os.path.join(os.path.pardir, 'README.md')
with open(README_path, 'r') as f:
    README = f.read()

__doc__ = README