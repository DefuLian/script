bin = E:/sync_cs_code/libfm-1.42.src/bin
negt = 10
comma:=,
empty:=
space:= $(empty) $(empty)
all: test.txt.N* evaluation$(negt).txt

#train.txt test.txt: ratings.tsv
#	ProcRec -rating=ratings.tsv -ratio=0.8 -user_thres=9 -item_thres=9

train.txt.neg%: train.txt test.txt
	DataProc -module=generate -train=train.txt -test=test.txt -negTimes=$(negt)

user_rel.txt item_rel.txt: train.txt test.txt
	DataProc -module=generate -train=train.txt -test=test.txt -negTimes=$(negt)
	
test.txt.N*: train.txt test.txt
	DataProc -module=generate -train=train.txt -test=test.txt -negTimes=-1
	
user_rel.xt: user_feature.txt
	converter.exe -ifile $< -ofilex user_rel.x -ofiley user_rel.y
	transpose.exe -ifile user_rel.x -ofile $@
	
item_rel.xt: item_feature.txt	
	converter.exe -ifile $< -ofilex item_rel.x -ofiley item_rel.y
	transpose.exe -ifile item_rel.x -ofile $@
	
train.xt train.y: train.txt.neg$(negt)
	cut -f1 $< > user_rel.train
	cut -f2 $< > item_rel.train
	cut -f3 $< > train.libfm
	converter.exe -ifile train.libfm -ofilex train.x -ofiley train.y
	transpose.exe -ifile train.x -ofile train.xt
	
eval%.txt: test.txt.N% train.xt train.y user_rel.xt item_rel.xt
	cut -f1 $< > user_rel.test
	cut -f2 $< > item_rel.test
	cut -f3 $< > test.libfm
	converter.exe -ifile test.libfm -ofilex test.x -ofiley test.y
	transpose.exe -ifile test.x -ofile test.xt
	libfm -task c -train train -test test -dim '0,0,50' --relation user_rel,item_rel --out pred.txt -iter 50
	DataProc -module=evaluation -mode=quick -test=$< -pred=pred.txt -result=$@
	
	
user_feature.txt:user_rel.txt
	DataProc -module=merge -features=$(subst $(space),$(comma),$^) -merged=$@

item_feature.txt:item_rel.txt
	DataProc -module=merge -features=$(subst $(space),$(comma),$^) -merged=$@

evaluation%.txt:$(patsubst test.txt.N%, eval%.txt, $(wildcard test.txt.N*))
	python E:/bin/aggregate.py $@ $(filter eval%.txt, $^)