#!/bin/bash
echo "prepare list"

ls ../datasets/fake/* | cut -d "/" -f 2- > fake_all_list.txt
grep -rnE [8-9].jpg fake_all_list.txt | cut -d : -f 2- > fake_data_test_list.txt
grep -rnE [0-3].jpg fake_all_list.txt | cut -d : -f 2- > fake_data_train_list.txt 
grep -rnE [4-7].jpg fake_all_list.txt | cut -d : -f 2- > fake_data_val_list.txt 

ls ../datasets/fake/* | cut -d "/" -f 2- > real_all_list.txt
grep -rnE [8-9].jpg real_all_list.txt | cut -d : -f 2- > real_data_test_list.txt
grep -rnE [0-3].jpg real_all_list.txt | cut -d : -f 2- > real_data_train_list.txt 
grep -rnE [4-7].jpg real_all_list.txt | cut -d : -f 2- > real_data_val_list.txt 

echo "prepare labels"
cnt=1
while(( $cnt<=400 ))
do
    let "cnt++"
    echo "1">> real_labels_train_list.txt
    echo "1">> real_labels_val_list.txt
    echo "0">> fake_labels_train_list.txt
    echo "0">> fake_labels_val_list.txt
done
cnt=1
while(( $cnt<=200 ))
do
    let "cnt++"
    echo "1">> real_labels_test_list.txt
    echo "0">> fake_labels_test_list.txt
done
