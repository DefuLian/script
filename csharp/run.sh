#!/bin/bash
reg_u=0.0025
reg_i=0.0025
reg_j=0.00025
learn_rate=0.05
m=122
if [ -e log1.txt ]; then rm log1.txt; fi;
for reg_u in '0.001' '0.01' '0.05' '0.1' '0.5' '1' '10'; do
	item_recommendation.exe --training-file=train.txt --test-file=test.txt --no-id-mapping --recommender=BPRMF --recommender-options="num_factors=50 reg_u=$reg_u reg_i=$reg_i reg_j=$reg_j learn_rate=$learn_rate num_iter=50" >> log1.txt
done
reg_u=`paste <(grep 'reg_u' log1.txt | sed -e 's/.*reg_u=\([0-9\.]\+\).*/\1/') <(grep '^[0-9]' log1.txt |cut -f$m ) | awk 'BEGIN{p=-1}{if(p<$2) {p=$2; v=$1}}END{print v}'`
cat log1.txt > log.txt
rm log1.txt

for reg_i in '0.001' '0.01' '0.05' '0.1' '0.5' '1'; do
	item_recommendation.exe --training-file=train.txt --test-file=test.txt --no-id-mapping --recommender=BPRMF --recommender-options="num_factors=50 reg_u=$reg_u reg_i=$reg_i reg_j=$reg_j learn_rate=$learn_rate num_iter=50" >> log1.txt
done
reg_i=`paste <(grep 'reg_i' log1.txt | sed -e 's/.*reg_i=\([0-9\.]\+\).*/\1/') <(grep '^[0-9]' log1.txt |cut -f$m ) | awk 'BEGIN{p=-1}{if(p<$2) {p=$2; v=$1}}END{print v}'`
cat log1.txt >> log.txt
rm log1.txt

for learn_rate in '0.001' '0.01' '0.1' '0.5'; do
	item_recommendation.exe --training-file=train.txt --test-file=test.txt --no-id-mapping --recommender=BPRMF --recommender-options="num_factors=50 reg_u=$reg_u reg_i=$reg_i reg_j=$reg_j learn_rate=$learn_rate num_iter=50" >> log1.txt
done
learn_rate=`paste <(grep 'learn_rate' log1.txt | sed -e 's/.*learn_rate=\([0-9\.]\+\).*/\1/') <(grep '^[0-9]' log1.txt |cut -f$m ) | awk 'BEGIN{p=-1}{if(p<$2) {p=$2; v=$1}}END{print v}'`
cat log1.txt >> log.txt
rm log1.txt

echo "$reg_u,$reg_i,$learn_rate" > rec_metric.txt
#grep '^[0-9]' log.txt | awk 'BEGIN{p=-1}{if(p<$81){p=$81;l=$0}}END{print l}' | awk '{for(i=2;i<=41;i++) printf $i"\t"; print '\n'; for(i=42;i<=81;i++) printf $i"\t"; print '\n'}' >> prec_recall.txt
grep '^[0-9]' log.txt | awk -v m=$m 'BEGIN{p=-1}{if(p<$m){p=$m;l=$0}}END{print l}' >> rec_metric.txt
mv log.txt log_bprmf.txt
mv rec_metric.txt rec_metric_bprmf.txt
