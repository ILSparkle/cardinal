.PHONY: quality style test

check_dirs := src

quality:
	ruff check $(check_dirs)
	ruff format --check $(check_dirs)

style:
	ruff check $(check_dirs) --fix
	ruff format $(check_dirs)

test:
	pytest tests/
