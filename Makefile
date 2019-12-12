venv:
	virtualenv venv && . venv/bin/activate && pip install -r requirements.lock

test: venv
	python setup.py install && pytest -s --cov pyoptimize --pdb tests

publish:
	rm -rf dist && python setup.py sdist bdist_wheel && \
	twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
