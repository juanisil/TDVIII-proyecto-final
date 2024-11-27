install:
	python3 -m pip install -r requirements.txt

test:
	python3 -m unittest discover ./tests -v

viz:
	cd gui/p2v_viz && npm run dev

create_conda_env:
	conda env create -f environment.yml

build_docs:
	cd docs && make html && make markdown

web_docs:
	open http://localhost:8000 && python3 -m http.server 8000 --directory docs/build/html
