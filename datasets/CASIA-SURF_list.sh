#!/bin/bash
rm -rf real_*
rm -rf fake_*

echo "prepare list"
# 8000 8000 4000
grep -rnE "fake.*depth" train_list.txt |cut -d " " -f 2 > _fake_train_depth_list.txt
sed -n '1,8000p' _fake_train_depth_list.txt > fake_data_train_list.txt 
sed -n '8001,16000p' _fake_train_depth_list.txt > fake_data_val_list.txt 
sed -n '16001,20000p' _fake_train_depth_list.txt > fake_data_test_list.txt
# 3200 3200 1600
grep -rnE "real.*depth" train_list.txt |cut -d " " -f 2 > _real_train_depth_list.txt
sed -n '1,3200p' _real_train_depth_list.txt > real_data_train_list.txt 
sed -n '3201,6400p' _real_train_depth_list.txt > real_data_val_list.txt 
sed -n '6401,8000p' _real_train_depth_list.txt > real_data_test_list.txt

echo "prepare labels"
cnt=1
while(( $cnt<=8000 ))
do
    echo "0">> fake_labels_train_list.txt
    echo "0">> fake_labels_val_list.txt
    if (($cnt <=4000 ))
    then
        echo "0">> fake_labels_test_list.txt
    fi
    let "cnt++"
done

cnt=1
while(( $cnt<=3200 ))
do
    echo "1">> real_labels_train_list.txt
    echo "1">> real_labels_val_list.txt
    if (($cnt <=1600 ))
    then
        echo "1">> real_labels_test_list.txt
    fi
    let "cnt++"
done
