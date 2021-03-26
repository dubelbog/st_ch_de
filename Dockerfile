FROM python:3

ADD fairseq /root/fairseq/
ADD test.py /

RUN apt-get update
RUN apt-get --yes install libsndfile1
RUN apt --yes install git-all
RUN apt --yes install ffmpeg

RUN source

RUN pip install pandas
RUN pip install soundfile
RUN pip install numpy
RUN pip install matplotlib
RUN pip install pandas
RUN pip install pydub
RUN pip install sacrebleu
RUN pip install torch
RUN pip install torchaudio
RUN pip install sentencepiece

#install fairseq over repo
RUN git clone https://github.com/pytorch/fairseq
RUN pip install --editable ./fairseq/

ENV PYTHONPATH "${PYTHONPATH}:root/fairseq"

CMD [ "sh", "-c","python ./test.py $TEST"]
#CMD ["sh", "-c", "python ./DidMain.py  $TRAIN $TEST $MODEL $EPOCHS $BSIZE"]
#CMD python DidMain.py $TRAIN $TEST $MODEL