

STEP 1 - Download IWSLT 2016 German–English parallel corpus and extract it to corpora/ folder.


wget -qO- --show-progress https://wit3.fbk.eu/archive/2016-01//texts/de/en/de-en.tgz | tar xz; mv de-en corpora


STEP 2 - Run train.py (see options in train.py) ---> logdir/

STEP 3 - Run test.py and see results in results/ folder

