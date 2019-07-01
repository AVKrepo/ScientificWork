#!/bin/bash
echo "transforming image $1 to output.png"

../ppng2pan $1 image.pan
../prgb2hsl image.pan data1.pan
../pgetband 0 data1.pan data44.pan

# ../pvisu data44.pan # not bad!

../pgetband 1 data1.pan data45.pan

# ../pvisu data45.pan # almost black

../pgetband 2 data1.pan data36.pan

#../pvisu data36.pan # almost original (lightness???)

../pdilatation 1 5 data36.pan data37.pan

#../pvisu data36.pan # before dilatation
#../pvisu data37.pan # after dilatation

../perosion 1 5 data37.pan data38.pan

../pvisu data38.pan # after erosion

../pmeanvalue data38.pan data41.pan # data41.pan is a float value
../psub data36.pan data38.pan data40.pan
../paddval data41.pan data40.pan data42.pan

#../pvisu data40.pan # after substraction
#../pvisu data42.pan # after adding mean value (they are almost identical?)

../pimg2imc 4 data44.pan data45.pan data42.pan data43.pan

#../pvisu data43.pan # result

../phsl2rgb data43.pan out7.pan
../ppan2png out7.pan output.png

#../ppan2png data1.pan data1.pan.png
#../ppan2ps data36.pan data36.pan.png
#../ppan2ppm data37.pan data37.pan.png
#../ppan2png data38.pan data38.pan.png
#../ppan2png data40.pan data40.pan.png
#../ppan2png data41.pan data41.pan.png
#../ppan2png data42.pan data42.pan.png
#../ppan2png data43.pan data44.pan.png
#../ppan2png data45.pan data45.pan.png
#../ppan2png data44.pan data44.pan.png

rm image.pan data1.pan data36.pan data37.pan data38.pan data40.pan data41.pan data42.pan data43.pan data44.pan data45.pan out7.pan


echo "Done!"
