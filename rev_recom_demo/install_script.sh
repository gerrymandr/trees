# Conda configuration
conda config --set always_yes yes --set auto_update_conda false
            
# install dependencies in conda env
conda install pytest pytest-cov
conda install codecov
python setup.py install
conda install shapely==1.6.4
            
# run tests
echo "backend: Agg" > "matplotlibrc"
pytest -v --cov=gerrychain --junitxml=test-reports/junit.xml tests
