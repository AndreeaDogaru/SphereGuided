cd data

wget http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip
unzip SampleSet.zip
mv "SampleSet/MVS Data/" dtu_eval
rm -r SampleSet
rm SampleSet.zip

wget http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip
unzip Points.zip -d dtu_eval
rm Points.zip