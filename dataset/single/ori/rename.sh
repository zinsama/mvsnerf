a=0
for num in {1..60..1}
do
mv $num.png `expr $num + 60`.png
done