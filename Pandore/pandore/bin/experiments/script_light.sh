#!/bin/bash

pandore_path="/home/avk/study/1C/Diplom/Pandore/pandore1/pandore"
operators_path="/home/avk/study/1C/Diplom/Pandore/pandore1/bin/"

echo "Before"
# OUTPUT1=$(sh $pandore_path)
echo "After"
mkdir 1

echo "transforming image $1 to $2"
echo "$operators_path"

../pjpeg2pan $1 image.pan
../prgb2hsl image.pan data1.pan
../pgetband 0 data1.pan data44.pan
../pgetband 1 data1.pan data45.pan
../pgetband 2 data1.pan data36.pan
../pdilatation 1 5 data36.pan data37.pan
../perosion 1 5 data37.pan data38.pan
../pmeanvalue data38.pan data41.pan
../psub data36.pan data38.pan data40.pan
../paddval data41.pan data40.pan data42.pan
../pimg2imc 4 data44.pan data45.pan data42.pan data43.pan
../phsl2rgb data43.pan out7.pan
../ppan2png out7.pan $2

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
