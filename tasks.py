import sys

import invoke

@invoke.task
def clean(context):
    '''Clean doc and build artifacts
    '''
    context.run('rm -rf coverage.xml')
    context.run('rm -rf htmlcov')
    context.run('rm -rf doc/build')
    context.run('rm -rf build')
    context.run('rm -rf dist')
    context.run('rm -rf *.egg-info')
    context.run('rm -rf .coverage')
    context.run('rm -rf .mypy_cache')
    context.run('rm -rf .pytest_cache')
    context.run('rm -rf .hypothesis')
    context.run('rm -rf .ipynb_checkpoints')


@invoke.task
def test(context):
    context.run(f'pytest test')

@invoke.task(pre=(clean,))
def build(context):
    '''Build packages
    '''
    # NOTE: must pip install build
    context.run(f'{sys.executable} -m build')


@invoke.task(pre=(build,), post=(clean,))
def release(context):
    context.run('twine upload dist/*')

