"""
Setup script para instalação do sistema DSGE-RL.

Para instalar em modo desenvolvimento:
    pip install -e .

Para instalar normalmente:
    pip install .
"""

from setuptools import setup, find_packages
import os

# Ler README
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Ler requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name='sistema-dsge-rl',
    version='1.0.0',
    description='Sistema de Previsão Econométrica com DSGE e Reinforcement Learning para Mercado Imobiliário',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='Sistema DSGE-RL Team',
    author_email='',
    url='https://github.com/cbaracho200/sistema-DSGE-RL',
    license='MIT',

    # Pacotes
    packages=find_packages(where='src'),
    package_dir={'': 'src'},

    # Dependências
    install_requires=read_requirements(),

    # Dependências extras
    extras_require={
        'dev': [
            'pytest>=7.3.0',
            'pytest-cov>=4.1.0',
            'black>=23.3.0',
            'flake8>=6.0.0',
            'ipykernel>=6.0.0',
        ],
        'notebooks': [
            'jupyter>=1.0.0',
            'ipython>=8.12.0',
            'notebook>=6.5.0',
            'plotly>=5.14.0',
        ],
    },

    # Metadados
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Office/Business :: Financial',
    ],

    keywords='forecasting econometrics dsge time-series machine-learning real-estate',

    python_requires='>=3.8',

    # Entry points (se houver comandos CLI)
    entry_points={
        'console_scripts': [
            # 'dsge-rl=pipeline:main',  # Descomentar se criar CLI
        ],
    },

    # Incluir arquivos adicionais
    include_package_data=True,
    package_data={
        '': ['*.yaml', '*.yml', '*.md'],
    },

    # Testes
    test_suite='tests',
    tests_require=['pytest>=7.3.0'],

    zip_safe=False,
)
