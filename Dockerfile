FROM Python
ADD . /dmd-energy-markets
RUN pip install /dmd-energy-markets/requirements.txt
RUN pip install /dmd-energy-markets