.PHONY: all clean

all: venv37pt14cu10/bin/activate venv37pt14cu10/lib/python3.7/site-packages/torch

venv37pt14cu10/bin/activate:
	#test -d venv37pt14cu10 || virtualenv -p python3.6 venv37pt14cu10
	test -d venv37pt14cu10 || virtualenv -p python3.7 venv37pt14cu10
	. venv37pt14cu10/bin/activate; pip install -r requirements.txt; pip install matplotlib;
	touch venv37pt14cu10/bin/activate

venv37pt14cu10/lib/python3.7/site-packages/torch: venv37pt14cu10/bin/activate
	#### For cuda10.1
	#. venv37pt14cu10/bin/activate; pip install pip --upgrade; pip3 install torch torchvision
	#### For cuda10.0
	### For python3.6
	#. venv37pt14cu10/bin/activate; pip install pip --upgrade; \
    #    pip3 install https://download.pytorch.org/whl/cu100/torch-1.4.0%2Bcu100-cp36-cp36m-linux_x86_64.whl; \
    #        pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.5.0%2Bcu100-cp36-cp36m-linux_x86_64.whl
	### For python3.7
	. venv37pt14cu10/bin/activate; pip install pip --upgrade; \
        pip3 install https://download.pytorch.org/whl/cu100/torch-1.4.0%2Bcu100-cp37-cp37m-linux_x86_64.whl; \
            pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.5.0%2Bcu100-cp37-cp37m-linux_x86_64.whl
	### see https://pytorch.org/ or
	### https://download.pytorch.org/whl/cu100/torch_stable.html or
	### https://download.pytorch.org/whl/cu92/torch_stable.html

clean:
	rm -fr venv37pt14cu10
	find -iname "*.pyc" -delete
