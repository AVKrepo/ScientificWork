

all:
	g++ -I/usr/include/opencv4 binarizewolfjolion.cpp -o binarizewolfjolion `pkg-config opencv4 --libs` -lstdc++

clean:
	rm -f binarizewolfjolion

test:
	./binarizewolfjolion -k 0.6 sample.jpg _result.jpg


package:	clean
	rm -f x.jpg
	tar cvfz binarizewolfjolionopencv.tgz *

