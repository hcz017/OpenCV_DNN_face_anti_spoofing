#!/bin/bash
rm -rf real_*
rm -rf fake_*

echo "prepare list"
grep -rnE "real.*depth" train_list.txt |cut -d " " -f 2 > real_train_depth_list.txt
grep -rnE "fake.*depth" train_list.txt |cut -d " " -f 2 > fake_train_depth_list.txt

sed -n '1,800p' real_train_depth_list.txt > real_data_train_list.txt 
sed -n '1001,1800p' real_train_depth_list.txt > real_data_val_list.txt 
sed -n '2001,2400p' real_train_depth_list.txt > real_data_test_list.txt

sed -n '1,800p' fake_train_depth_list.txt > fake_data_train_list.txt 
sed -n '1001,1800p' fake_train_depth_list.txt > fake_data_val_list.txt 
sed -n '2001,2400p' fake_train_depth_list.txt > fake_data_test_list.txt


echo "prepare labels"
cnt=1
while(( $cnt<=800 ))
do
    let "cnt++"
    echo "1">> real_labels_train_list.txt
    echo "1">> real_labels_val_list.txt
    echo "0">> fake_labels_train_list.txt
    echo "0">> fake_labels_val_list.txt
done
cnt=1
while(( $cnt<=400 ))
do
    let "cnt++"
    echo "1">> real_labels_test_list.txt
    echo "0">> fake_labels_test_list.txt
done
