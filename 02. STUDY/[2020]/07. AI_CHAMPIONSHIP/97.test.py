import re

file_train_mod = "C:/Users/wook1/Documents/WAI/2020/02.K스타트업_빅데이터경진대회/개발/data/train.csv"
file_train_rep ="C:/Users/wook1/Documents/WAI/2020/02.K스타트업_빅데이터경진대회/개발//data/prep/train_rep_sub.csv"
file_train_rep_test ="C:/Users/wook1/Documents/WAI/2020/02.K스타트업_빅데이터경진대회/개발//data/prep/train_rep_sub_test.csv"
with open(file_train_rep, "w") as csv_file:
    line_count = 0
    file_source = open(file_train_mod, "r")
    columns = str(file_source.readline()).strip().split(",")
    csv_file.write(",".join(columns) + "\n")
    flag_while_loop = True
    set_check = set()
    while flag_while_loop:
        line = str(file_source.readline()).strip()
        if not line:
            flag_while_loop = False
            continue


import csv
with open(file_train_mod) as csvfile:
    train_file = csv.reader(csvfile)
    error_list = []
    for idx, row in enumerate(train_file):
        if len(row) != 24:
            print("{0} 행의 데이터의 길이는 {1}입니다.".format(idx,len(row)))
            error_list.append(idx)
        #print(", ".join(row))
len(error_list)


PATH = "C:/Users/wook1/Documents/WAI/2020/02.K스타트업_빅데이터경진대회/개발/"
file_train_mod = PATH + "data/train.csv"
file_train_rep =PATH + "/data/prep/train_rep.csv"
f_read = open(file_train_mod, 'r', encoding="utf-8")
lines = f_read.readlines()
f_read.close()
read_index = [1303189, 2502332, 2728197,6320085, 7512591,  9012823, 9587574]
for i in read_index:
    print(lines[i].split(","))
    print(len(lines[i]))

with open(file_train_rep, 'a', encoding="cp949") as csv_file:
    line_count = 0
    file_source = open(file_train_mod, "r")
    columns = str(file_source.readline()).strip().split(",")
    csv_file.write(",".join(columns) + "\n")
    flag_while_loop = True
    set_check = set()
    for i in read_index:

        list_values = lines[i].split(",")
        column_count = len(list_values)

        # 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 54, 55, 57, 70
        # if column_count != 70:
        #     continue
        print(list_values)
        try:
            ord_no = list_values[0]
            ci_seq = list_values[1]
            mem_no = list_values[2]
            dvc_id = list_values[3]
            shop_no = list_values[4]
            shop_owner_no = list_values[5]
            rgn1_cd = list_values[6]
            rgn2_cd = list_values[7]
            rgn3_cd = list_values[8]
            # comma 를 주의해서 추출해야 하는 구간
            ord_msg = list()
            ord_tm = ""
            item_name = list()
            flag_split = True
            for idx in range(9, column_count - 12):
                word = list_values[idx]
                if re.match("\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", word):
                    flag_split = False
                    ord_tm = word
                elif re.match("\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d{3}", word):
                    flag_split = False
                    ord_tm = word
                else:
                    if flag_split:
                        ord_msg.append(word)
                        #print(ord_msg)
                    else:
                        item_name.append(word)
                        #print(item_name)
            #print(ord_msg)
            #print(ord_tm)
            #print(item_name)
            item_quantity = list_values[-12]
            cpn_use_cnt = list_values[-11]
            ord_price = list_values[-10]
            purch_method_cd = list_values[-9]
            review_yn = list_values[-8]
            rating = list_values[-7]
            review_created_tm = list_values[-6]
            image_review_yn = list_values[-5]
            delivery_yn = list_values[-4]
            ord_prog_cd = list_values[-3]
            ord_date = list_values[-2]
            ord_dt = list_values[-1]
            print()
            csv_file.write(str(ord_no) + ",")
            csv_file.write(str(ci_seq) + ",")
            csv_file.write(str(mem_no) + ",")
            csv_file.write(str(dvc_id) + ",")
            csv_file.write(str(shop_no) + ",")
            csv_file.write(str(shop_owner_no) + ",")
            csv_file.write(str(rgn1_cd) + ",")
            csv_file.write(str(rgn2_cd) + ",")
            csv_file.write(str(rgn3_cd) + ",")
            csv_file.write(str("|".join(ord_msg)) + ",")
            csv_file.write(str(ord_tm) + ",")
            csv_file.write(str("|".join(item_name)) + ",")
            csv_file.write(str(item_quantity) + ",")
            csv_file.write(str(cpn_use_cnt) + ",")
            csv_file.write(str(ord_price) + ",")
            csv_file.write(str(purch_method_cd) + ",")
            csv_file.write(str(review_yn) + ",")
            csv_file.write(str(rating) + ",")
            csv_file.write(str(review_created_tm) + ",")
            csv_file.write(str(image_review_yn) + ",")
            csv_file.write(str(delivery_yn) + ",")
            csv_file.write(str(ord_prog_cd) + ",")
            csv_file.write(str(ord_date) + ",")
            csv_file.write(str(ord_dt))
            #csv_file.write("\n")

            line_count += 1
        except:
            print(line)
            print(line_count)




