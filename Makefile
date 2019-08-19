test:
	python -m pytest -s tests --cov=pyoptimize --pdb

publish:
	rm -rf dist && python setup.py sdist bdist_wheel && \
	twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
