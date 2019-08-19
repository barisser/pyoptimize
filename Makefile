test:
	python -m pytest -s tests --cov=pyoptimize --pdb

publish:
	twine upload --repository-url https://upload.pypi.org/legacy dist/*
