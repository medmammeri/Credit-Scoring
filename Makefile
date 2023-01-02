.PHONY: install
install:
	pip install -e '.[dev]'

.PHONY: push
push:
	git add .
	git commit -m'update'
	git push

.PHONY: install_doc
install_doc:
	pip install -r requirements/doc