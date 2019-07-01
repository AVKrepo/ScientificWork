#!/bin/bash
echo "transforming image $1 to $2"

../pjpeg2pan $1 image.pan
../prgb2hsl image.pan data1.pan
../pgetband 0 data1.pan data44.pan

# ../pvisu data44.pan # not bad!

../pgetband 1 data1.pan data45.pan

# ../pvisu data45.pan # almost black

../pgetband 2 data1.pan data36.pan

#../pvisu data36.pan # almost original (lightness???)

../pdilatation 1 5 data36.pan data37.pan
##../pdilatation 1 15 data36.pan data37_.pan

#../pvisu data37.pan # after dilatation
#../pvisu data37_.pan # after dilatation


#../pvisu data36.pan # before dilatation
#../pvisu data37.pan # after dilatation

../perosion 1 5 data37.pan data38.pan
##../perosion 1 15 data37_.pan data38_.pan

######################../pvisu data38.pan # after erosion
##../pvisu data38_.pan # after erosion

../pmeanvalue data38.pan data41.pan # data41.pan is a float value
##../pmeanvalue data38_.pan data41_.pan # data41.pan is a float value
../psub data36.pan data38.pan data40.pan
##../psub data36.pan data38_.pan data40_.pan
../paddval data41.pan data40.pan data42.pan
###cp data40.pan data42.pan
##../paddval data41_.pan data40_.pan data42_.pan

#../pvisu data40.pan # after substraction
#../pvisu data42.pan # after adding mean value (they are almost identical?)
#../pvisu data42_.pan

../pnormalization 0 255 data42.pan result.pan

../pimg2imc 4 data44.pan data45.pan result.pan data43.pan

#../pvisu data43.pan # result

../phsl2rgb data43.pan out7.pan
../ppan2png out7.pan $2



rm ./*.pan


echo "Done!"
