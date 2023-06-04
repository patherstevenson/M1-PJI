PROJECT=Segmentation-PJI
AUTHOR=Stevenson Pather
PY3=python3
PYTHONPATH=./src
export PYTHONPATH
SPHINXBUILD=sphinx-build
CONFIGPATH=.
SOURCEDOC=sourcedoc
DOC=doc
SCRIPT=install_lib.sh
MAIN=src/main.py

lib:
	chmod +x $(SCRIPT)
	./$(SCRIPT)

clean:
	rm -f *~ */*~
	rm -rf __pycache__ src/__pycache__
	rm -rf $(DOC)
	rm -f $(PROJECT).zip

doc: author
	$(SPHINXBUILD) -c $(CONFIGPATH) -b html $(SOURCEDOC) $(DOC)

archive: clean
	zip -r $(PROJECT).zip .

author:
	sed -i -e 's/^project =.*/project = "Unsupervised image segmentation algorithms applied to object detection"/g' conf.py
	sed -i -e 's/^copyright =.*/copyright = "2023, $(AUTHOR), Univ. Lille"/g' conf.py
	sed -i -e 's/^author =.*/author = "$(AUTHOR)"/g' conf.py

.PHONY: clean doc archive author lib
