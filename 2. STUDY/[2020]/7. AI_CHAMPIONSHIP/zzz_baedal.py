import warnings
import sys
import re
import pandas
import numpy

file_train = '/opt/analysis/use-cases/whiz-apollon-docker/dist/abuse/train.csv'
file_train_rep = '/opt/analysis/use-cases/whiz-apollon-docker/dist/abuse/train_rep.csv'
file_valid = '/opt/analysis/use-cases/whiz-apollon-docker/dist/abuse/validation.csv'

file_train_base = '/opt/analysis/use-cases/whiz-apollon-docker/dist/abuse/train_base.csv'
file_train_ord_msg = '/opt/analysis/use-cases/whiz-apollon-docker/dist/abuse/train_ord_msg.csv'
file_train_item_name = '/opt/analysis/use-cases/whiz-apollon-docker/dist/abuse/train_item_name.csv'

file_master_codes = '/opt/analysis/use-cases/whiz-apollon-docker/dist/abuse/master_codes.csv'

file_pre_processed = '/opt/analysis/use-cases/whiz-apollon-docker/dist/abuse/source_data.csv'

def step_01_clean_csv():
    with open(file_train_rep, 'w') as csv_file:
        line_count = 0
        file_source = open(file_train, 'r')
        columns = str(file_source.readline()).strip().split(',')
        csv_file.write(','.join(columns) + '\n')
        flag_while_loop = True
        set_check = set()
        accm_line = ''
        prev_line = ''
        flag_accm = False
        while flag_while_loop:
            line = str(file_source.readline()).strip()
            if not line:
                flag_while_loop = False
                continue
            list_values = line.split(',')

            ord_dt = list_values[-1]
            if re.match('\d{4}-\d{2}-\d{2}', ord_dt):
                if flag_accm:
                    accm_line = prev_line + ' ' + line
                    prev_line = ''
                    print(accm_line + '\n\n')
                    flag_accm = False
                else:
                    accm_line = line
            else:
                print(line)
                prev_line = line
                flag_accm = True

            list_values = accm_line.split(',')
            column_count = len(list_values)
            # 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 54, 55, 57, 70
            # if column_count != 70:
            #     continue

            # print(list_values)

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
            ord_tm = ''
            item_name = list()
            flag_split = True
            for idx in range(9, column_count - 9):
                word = list_values[idx]
                if re.match('\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', word):
                    flag_split = False
                    ord_tm = word
                else:
                    if flag_split:
                        ord_msg.append(word)
                    else:
                        item_name.append(word)

            # print(ord_msg)
            # print(ord_tm)
            # print(item_name)

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

            csv_file.write(str(ord_no) + ',')
            csv_file.write(str(ci_seq) + ',')
            csv_file.write(str(mem_no) + ',')
            csv_file.write(str(dvc_id) + ',')
            csv_file.write(str(shop_no) + ',')
            csv_file.write(str(shop_owner_no) + ',')
            csv_file.write(str(rgn1_cd) + ',')
            csv_file.write(str(rgn2_cd) + ',')
            csv_file.write(str(rgn3_cd) + ',')

            csv_file.write(str('|'.join(ord_msg)) + ',')
            csv_file.write(str(ord_tm).split('.')[0] + ',')
            csv_file.write(str('|'.join(item_name)) + ',')

            csv_file.write(str(item_quantity) + ',')
            csv_file.write(str(cpn_use_cnt) + ',')
            csv_file.write(str(ord_price) + ',')
            csv_file.write(str(purch_method_cd) + ',')
            csv_file.write(str(review_yn) + ',')
            csv_file.write(str(rating) + ',')
            csv_file.write(str(review_created_tm).split('.')[0] + ',')
            csv_file.write(str(image_review_yn) + ',')
            csv_file.write(str(delivery_yn) + ',')
            csv_file.write(str(ord_prog_cd) + ',')
            csv_file.write(str(ord_date).split('.')[0] + ',')
            csv_file.write(str(ord_dt))
            csv_file.write('\n')

            line_count += 1
            if line_count % 10000 == 0:
                print(str(line_count).zfill(10), flush=True)
                pass

        print(str(line_count).zfill(10), flush=True)


def step_02_split_file():
    df_base = pandas.read_csv(file_train_rep)

    df_ord_msg = df_base[['shop_no', 'ord_dt', 'ord_msg']]
    print(df_ord_msg.head())
    print(df_ord_msg['ord_msg'].unique())

    del df_base['ord_msg']

    df_item_name = df_base[['shop_no', 'ord_dt', 'item_name']]
    print(df_item_name.head())
    print(df_item_name['item_name'].unique())

    del df_base['item_name']

    print(df_base.head())
    print(df_base['ord_dt'].unique())

    df_base.to_csv(file_train_base, index=None)
    df_ord_msg.to_csv(file_train_ord_msg, index=None)
    df_item_name.to_csv(file_train_item_name, index=None)


def numpy_std(x): 
    return numpy.std(x)


def step_03_generate_base_code():
    df_base = pandas.read_csv(file_train_base, nrows=200)
    df_base = pandas.read_csv(file_train_base)

    df_pre_processed = pandas.DataFrame()

    # ===============================================================================================================================================================
    # Code replace
    # ===============================================================================================================================================================
    list_code_names = list()
    list_codes = df_base['purch_method_cd'].unique()
    code_number = 0
    for purch_method_cd in list_codes:
        list_code_names.append({
            'COLUMN_NAME': 'purch_method_cd',
            'VALUE_ORIGINAL': purch_method_cd,
            'VALUE_REPLACED': 'CD_1_' + str(code_number).zfill(2)
        })
        df_base.loc[df_base['purch_method_cd'] == purch_method_cd, 'purch_method_cd'] = 'CD_1_' + str(code_number).zfill(2)
        code_number += 1
    print('purch_method_cd', df_base['purch_method_cd'].unique())

    list_codes = df_base['rgn1_cd'].unique()
    code_number = 0
    for rgn1_cd in list_codes:
        list_code_names.append({
            'COLUMN_NAME': 'rgn1_cd',
            'VALUE_ORIGINAL': rgn1_cd,
            'VALUE_REPLACED': 'CD_2_' + str(code_number).zfill(2)
        })
        df_base.loc[df_base['rgn1_cd'] == rgn1_cd, 'rgn1_cd'] = 'CD_2_' + str(code_number).zfill(2)
        code_number += 1
    print('rgn1_cd', df_base['rgn1_cd'].unique())

    list_codes = df_base['rgn2_cd'].unique()
    code_number = 0
    for rgn2_cd in list_codes:
        list_code_names.append({
            'COLUMN_NAME': 'rgn2_cd',
            'VALUE_ORIGINAL': rgn2_cd,
            'VALUE_REPLACED': 'CD_3_' + str(code_number).zfill(3)
        })
        df_base.loc[df_base['rgn2_cd'] == rgn2_cd, 'rgn2_cd'] = 'CD_3_' + str(code_number).zfill(3)
        code_number += 1
    print('rgn2_cd', df_base['rgn2_cd'].unique())

    list_codes = df_base['rgn3_cd'].unique()
    code_number = 0
    for rgn3_cd in list_codes:
        list_code_names.append({
            'COLUMN_NAME': 'rgn3_cd',
            'VALUE_ORIGINAL': rgn3_cd,
            'VALUE_REPLACED': 'CD_4_' + str(code_number).zfill(4)
        })
        df_base.loc[df_base['rgn3_cd'] == rgn3_cd, 'rgn3_cd'] = 'CD_4_' + str(code_number).zfill(4)
        code_number += 1
    print('rgn3_cd', df_base['rgn3_cd'].unique())

    list_codes = df_base['ord_prog_cd'].unique()
    code_number = 0
    for ord_prog_cd in list_codes:
        list_code_names.append({
            'COLUMN_NAME': 'ord_prog_cd',
            'VALUE_ORIGINAL': ord_prog_cd,
            'VALUE_REPLACED': 'CD_5_' + str(code_number)
        })
        df_base.loc[df_base['ord_prog_cd'] == ord_prog_cd, 'ord_prog_cd'] = 'CD_5_' + str(code_number)
        code_number += 1
    print('ord_prog_cd', df_base['ord_prog_cd'].unique())

    list_codes = df_base['review_yn'].unique()
    code_number = 0
    for review_yn in list_codes:
        list_code_names.append({
            'COLUMN_NAME': 'review_yn',
            'VALUE_ORIGINAL': review_yn,
            'VALUE_REPLACED': 'CD_6_' + str(code_number)
        })
        df_base.loc[df_base['review_yn'] == review_yn, 'review_yn'] = 'CD_6_' + str(code_number)
        code_number += 1
    print('review_yn', df_base['review_yn'].unique())

    list_codes = df_base['image_review_yn'].unique()
    code_number = 0
    for image_review_yn in list_codes:
        list_code_names.append({
            'COLUMN_NAME': 'image_review_yn',
            'VALUE_ORIGINAL': image_review_yn,
            'VALUE_REPLACED': 'CD_7_' + str(code_number)
        })
        df_base.loc[df_base['image_review_yn'] == image_review_yn, 'image_review_yn'] = 'CD_7_' + str(code_number)
        code_number += 1
    print('image_review_yn', df_base['image_review_yn'].unique())

    list_codes = df_base['delivery_yn'].unique()
    code_number = 0
    for delivery_yn in list_codes:
        list_code_names.append({
            'COLUMN_NAME': 'delivery_yn',
            'VALUE_ORIGINAL': delivery_yn,
            'VALUE_REPLACED': 'CD_8_' + str(code_number)
        })
        df_base.loc[df_base['delivery_yn'] == delivery_yn, 'delivery_yn'] = 'CD_8_' + str(code_number)
        code_number += 1
    print('delivery_yn', df_base['delivery_yn'].unique())

    df_codes = pandas.DataFrame(list_code_names)
    df_codes.to_csv(file_master_codes, index=None)
    print(df_codes.head(20))
    # ===============================================================================================================================================================

    # ===============================================================================================================================================================
    # Time extract
    # ===============================================================================================================================================================
    df_base['ord_hour'] = df_base['ord_tm'].str.slice(start=11, stop=13)
    df_base['ord_hour'] = df_base['ord_hour'].astype(int)

    df_base['review_created_tm']  = df_base['review_created_tm'].str.rjust(width=15, fillchar='0') 
    df_base['review_created_hour'] = df_base['review_created_tm'].str.slice(start=11, stop=13)
    # df_check = df_base[['review_created_hour', 'review_created_tm']]
    # print(df_check.head(30))
    df_base['review_created_hour'] = df_base['review_created_hour'].fillna(0)
    df_base['review_created_hour'] = df_base['review_created_hour'].astype(int)
    # ===============================================================================================================================================================

    # ===============================================================================================================================================================
    # ID count
    # ===============================================================================================================================================================
    # ci_seq 기준 건수 집계
    df_ci_seq = df_base[['shop_no', 'ord_dt', 'ci_seq', 'ord_no']]
    df_group_ci_seq = df_ci_seq.groupby(['shop_no', 'ord_dt', 'ci_seq']).agg(
        CNT_ci_seq = ('ord_no', 'count'),
    )
    df_group_ci_seq = df_group_ci_seq.sort_index(ascending=True)
    df_group_ci_seq = df_group_ci_seq.reset_index()
    df_summary_ci_seq = df_group_ci_seq.groupby(['shop_no', 'ord_dt']).agg(
        MAX_ci_seq = ('CNT_ci_seq', max),
        MIN_ci_seq = ('CNT_ci_seq', min),
        SUM_ci_seq = ('CNT_ci_seq', sum),
    )
    df_pre_processed = df_summary_ci_seq.copy()
    print(df_pre_processed.shape, df_pre_processed.head(10))

    # mem_no 기준 건수 집계
    df_mem_no = df_base[['shop_no', 'ord_dt', 'mem_no', 'ord_no']]
    df_group_mem_no = df_mem_no.groupby(['shop_no', 'ord_dt', 'mem_no']).agg(
        CNT_mem_no = ('ord_no', 'count'),
    )
    df_group_mem_no = df_group_mem_no.sort_index(ascending=True)
    df_group_mem_no = df_group_mem_no.reset_index()
    df_summary_mem_no = df_group_mem_no.groupby(['shop_no', 'ord_dt']).agg(
        MAX_mem_no = ('CNT_mem_no', max),
        MIN_mem_no = ('CNT_mem_no', min),
        SUM_mem_no = ('CNT_mem_no', sum),
        AVG_mem_no = ('CNT_mem_no', "mean"),
    )
    df_pre_processed = df_pre_processed.join(df_summary_mem_no, how='inner')
    print(df_pre_processed.shape, df_pre_processed.head(10))

    # dvc_id 기준 건수 집계
    df_dvc_id = df_base[['shop_no', 'ord_dt', 'dvc_id', 'ord_no']]
    df_group_dvc_id = df_dvc_id.groupby(['shop_no', 'ord_dt', 'dvc_id']).agg(
        CNT_dvc_id = ('ord_no', 'count'),
    )
    df_group_dvc_id = df_group_dvc_id.sort_index(ascending=True)
    df_group_dvc_id = df_group_dvc_id.reset_index()
    df_summary_dvc_id = df_group_dvc_id.groupby(['shop_no', 'ord_dt']).agg(
        MAX_dvc_id = ('CNT_dvc_id', max),
        MIN_dvc_id = ('CNT_dvc_id', min),
        SUM_dvc_id = ('CNT_dvc_id', sum),
        AVG_dvc_id = ('CNT_dvc_id', "mean"),
    )
    df_pre_processed = df_pre_processed.join(df_summary_dvc_id, how='inner')
    print(df_pre_processed.shape, df_pre_processed.head(10))
    # ===============================================================================================================================================================

    # ===============================================================================================================================================================
    # Continuous summary
    # ===============================================================================================================================================================
    # item_quantity 기준 건수 집계
    df_item_quantity = df_base[['shop_no', 'ord_dt', 'item_quantity', 'ord_no']]
    df_summary_item_quantity = df_item_quantity.groupby(['shop_no', 'ord_dt']).agg(
        MAX_item_quantity = ('item_quantity', max),
        MIN_item_quantity = ('item_quantity', min),
        SUM_item_quantity = ('item_quantity', sum),
        AVG_item_quantity = ('item_quantity', "mean"),
        STD_item_quantity = ('item_quantity', numpy_std),
    )
    df_pre_processed = df_pre_processed.join(df_summary_item_quantity, how='inner')
    print(df_pre_processed.shape, df_pre_processed.head(10))

    # cpn_use_cnt 기준 건수 집계
    df_cpn_use_cnt = df_base[['shop_no', 'ord_dt', 'cpn_use_cnt', 'ord_no']]
    df_summary_cpn_use_cnt = df_cpn_use_cnt.groupby(['shop_no', 'ord_dt']).agg(
        MAX_cpn_use_cnt = ('cpn_use_cnt', max),
        MIN_cpn_use_cnt = ('cpn_use_cnt', min),
        SUM_cpn_use_cnt = ('cpn_use_cnt', sum),
        AVG_cpn_use_cnt = ('cpn_use_cnt', "mean"),
        STD_cpn_use_cnt = ('cpn_use_cnt', numpy_std),
    )
    df_pre_processed = df_pre_processed.join(df_summary_cpn_use_cnt, how='inner')
    print(df_pre_processed.shape, df_pre_processed.head(10))

    # ord_price 기준 건수 집계
    df_ord_price = df_base[['shop_no', 'ord_dt', 'ord_price', 'ord_no']]
    df_summary_ord_price = df_ord_price.groupby(['shop_no', 'ord_dt']).agg(
        MAX_ord_price = ('ord_price', max),
        MIN_ord_price = ('ord_price', min),
        SUM_ord_price = ('ord_price', sum),
        AVG_ord_price = ('ord_price', "mean"),
        STD_ord_price = ('ord_price', numpy_std),
    )
    df_pre_processed = df_pre_processed.join(df_summary_ord_price, how='inner')
    print(df_pre_processed.shape, df_pre_processed.head(10))
    # ===============================================================================================================================================================

    # ===============================================================================================================================================================
    # Code count
    # ===============================================================================================================================================================
    # purch_method_cd 기준 건수 집계
    df_purch_method_cd = df_base[['shop_no', 'ord_dt', 'purch_method_cd', 'ord_no']]
    df_group_purch_method_cd = df_purch_method_cd.groupby(['shop_no', 'ord_dt', 'purch_method_cd']).agg(
        CNT_purch_method_cd = ('ord_no', 'count'),
    )
    df_group_purch_method_cd = df_group_purch_method_cd.sort_index(ascending=True)
    df_group_purch_method_cd = df_group_purch_method_cd.reset_index()
    df_summary_purch_method_cd = df_group_purch_method_cd.groupby(['shop_no', 'ord_dt']).agg(
        MAX_purch_method_cd = ('CNT_purch_method_cd', max),
        MIN_purch_method_cd = ('CNT_purch_method_cd', min),
        SUM_purch_method_cd = ('CNT_purch_method_cd', sum),
        AVG_purch_method_cd = ('CNT_purch_method_cd', "mean"),
    )
    df_pre_processed = df_pre_processed.join(df_summary_purch_method_cd, how='inner')
    print(df_pre_processed.shape, df_pre_processed.head(10))
    df_purch_method_cd['constants'] = 1
    df_transpose_purch_method_cd = pandas.pivot_table(df_purch_method_cd, values='constants', index=['shop_no', 'ord_dt'], columns=['purch_method_cd'], aggfunc=numpy.sum, fill_value=0)
    df_transpose_purch_method_cd.columns = [str(col) for col in df_transpose_purch_method_cd.columns.values]
    df_pre_processed = df_pre_processed.join(df_transpose_purch_method_cd, how='inner')
    print(df_pre_processed.shape, df_pre_processed.head(10))

    # review_yn 기준 건수 집계
    df_review_yn = df_base[['shop_no', 'ord_dt', 'review_yn', 'ord_no']]
    df_group_review_yn = df_review_yn.groupby(['shop_no', 'ord_dt', 'review_yn']).agg(
        CNT_review_yn = ('ord_no', 'count'),
    )
    df_group_review_yn = df_group_review_yn.sort_index(ascending=True)
    df_group_review_yn = df_group_review_yn.reset_index()
    df_summary_review_yn = df_group_review_yn.groupby(['shop_no', 'ord_dt']).agg(
        SUM_review_yn = ('CNT_review_yn', sum),
        AVG_review_yn = ('CNT_review_yn', "mean"),
    )
    df_pre_processed = df_pre_processed.join(df_summary_review_yn, how='inner')
    print(df_pre_processed.shape, df_pre_processed.head(10))
    df_review_yn['constants'] = 1
    df_transpose_review_yn = pandas.pivot_table(df_review_yn, values='constants', index=['shop_no', 'ord_dt'], columns=['review_yn'], aggfunc=numpy.sum, fill_value=0)
    df_transpose_review_yn.columns = [str(col) for col in df_transpose_review_yn.columns.values]
    df_pre_processed = df_pre_processed.join(df_transpose_review_yn, how='inner')
    print(df_pre_processed.shape, df_pre_processed.head(10))

    # rgn1_cd 기준 건수 집계
    df_rgn1_cd = df_base[['shop_no', 'ord_dt', 'rgn1_cd', 'ord_no']]
    df_group_rgn1_cd = df_rgn1_cd.groupby(['shop_no', 'ord_dt', 'rgn1_cd']).agg(
        CNT_rgn1_cd = ('ord_no', 'count'),
    )
    df_group_rgn1_cd = df_group_rgn1_cd.sort_index(ascending=True)
    df_group_rgn1_cd = df_group_rgn1_cd.reset_index()
    df_summary_rgn1_cd = df_group_rgn1_cd.groupby(['shop_no', 'ord_dt']).agg(
        MAX_rgn1_cd = ('CNT_rgn1_cd', max),
        MIN_rgn1_cd = ('CNT_rgn1_cd', min),
        SUM_rgn1_cd = ('CNT_rgn1_cd', sum),
        AVG_rgn1_cd = ('CNT_rgn1_cd', "mean"),
    )
    df_pre_processed = df_pre_processed.join(df_summary_rgn1_cd, how='inner')
    print(df_pre_processed.shape, df_pre_processed.head(10))
    df_rgn1_cd['constants'] = 1
    df_transpose_rgn1_cd = pandas.pivot_table(df_rgn1_cd, values='constants', index=['shop_no', 'ord_dt'], columns=['rgn1_cd'], aggfunc=numpy.sum, fill_value=0)
    df_transpose_rgn1_cd.columns = [str(col) for col in df_transpose_rgn1_cd.columns.values]
    df_pre_processed = df_pre_processed.join(df_transpose_rgn1_cd, how='inner')
    print(df_pre_processed.shape, df_pre_processed.head(10))

    # rgn2_cd 기준 건수 집계
    df_rgn2_cd = df_base[['shop_no', 'ord_dt', 'rgn2_cd', 'ord_no']]
    df_group_rgn2_cd = df_rgn2_cd.groupby(['shop_no', 'ord_dt', 'rgn2_cd']).agg(
        CNT_rgn2_cd = ('ord_no', 'count'),
    )
    df_group_rgn2_cd = df_group_rgn2_cd.sort_index(ascending=True)
    df_group_rgn2_cd = df_group_rgn2_cd.reset_index()
    df_summary_rgn2_cd = df_group_rgn2_cd.groupby(['shop_no', 'ord_dt']).agg(
        MAX_rgn2_cd = ('CNT_rgn2_cd', max),
        MIN_rgn2_cd = ('CNT_rgn2_cd', min),
        SUM_rgn2_cd = ('CNT_rgn2_cd', sum),
        AVG_rgn2_cd = ('CNT_rgn2_cd', "mean"),
    )
    df_pre_processed = df_pre_processed.join(df_summary_rgn2_cd, how='inner')
    print(df_pre_processed.shape, df_pre_processed.head(10))
    df_rgn2_cd['constants'] = 1
    df_transpose_rgn2_cd = pandas.pivot_table(df_rgn2_cd, values='constants', index=['shop_no', 'ord_dt'], columns=['rgn2_cd'], aggfunc=numpy.sum, fill_value=0)
    df_transpose_rgn2_cd.columns = [str(col) for col in df_transpose_rgn2_cd.columns.values]
    df_pre_processed = df_pre_processed.join(df_transpose_rgn2_cd, how='inner')
    print(df_pre_processed.shape, df_pre_processed.head(10))

    # rgn3_cd 기준 건수 집계
    df_rgn3_cd = df_base[['shop_no', 'ord_dt', 'rgn3_cd', 'ord_no']]
    df_group_rgn3_cd = df_rgn3_cd.groupby(['shop_no', 'ord_dt', 'rgn3_cd']).agg(
        CNT_rgn3_cd = ('ord_no', 'count'),
    )
    df_group_rgn3_cd = df_group_rgn3_cd.sort_index(ascending=True)
    df_group_rgn3_cd = df_group_rgn3_cd.reset_index()
    df_summary_rgn3_cd = df_group_rgn3_cd.groupby(['shop_no', 'ord_dt']).agg(
        MAX_rgn3_cd = ('CNT_rgn3_cd', max),
        MIN_rgn3_cd = ('CNT_rgn3_cd', min),
        SUM_rgn3_cd = ('CNT_rgn3_cd', sum),
        AVG_rgn3_cd = ('CNT_rgn3_cd', "mean"),
    )
    df_pre_processed = df_pre_processed.join(df_summary_rgn3_cd, how='inner')
    print(df_pre_processed.shape, df_pre_processed.head(10))
    df_rgn3_cd['constants'] = 1
    df_transpose_rgn3_cd = pandas.pivot_table(df_rgn3_cd, values='constants', index=['shop_no', 'ord_dt'], columns=['rgn3_cd'], aggfunc=numpy.sum, fill_value=0)
    df_transpose_rgn3_cd.columns = [str(col) for col in df_transpose_rgn3_cd.columns.values]
    df_pre_processed = df_pre_processed.join(df_transpose_rgn3_cd, how='inner')
    print(df_pre_processed.shape, df_pre_processed.head(10))
    # ===============================================================================================================================================================

    # ===============================================================================================================================================================
    # Hour count
    # ===============================================================================================================================================================
    # ord_hour 기준 건수 집계
    df_ord_hour = df_base[['shop_no', 'ord_dt', 'ord_hour', 'ord_no']]
    df_group_ord_hour = df_ord_hour.groupby(['shop_no', 'ord_dt', 'ord_hour']).agg(
        CNT_ord_hour = ('ord_no', 'count'),
    )
    df_group_ord_hour = df_group_ord_hour.sort_index(ascending=True)
    df_group_ord_hour = df_group_ord_hour.reset_index()
    df_summary_ord_hour = df_group_ord_hour.groupby(['shop_no', 'ord_dt']).agg(
        MAX_ord_hour = ('CNT_ord_hour', max),
        MIN_ord_hour = ('CNT_ord_hour', min),
        SUM_ord_hour = ('CNT_ord_hour', sum),
        AVG_ord_hour = ('CNT_ord_hour', "mean"),
    )
    df_pre_processed = df_pre_processed.join(df_summary_ord_hour, how='inner')
    print(df_pre_processed.shape, df_pre_processed.head(10))
    df_ord_hour['constants'] = 1
    df_transpose_ord_hour = pandas.pivot_table(df_ord_hour, values='constants', index=['shop_no', 'ord_dt'], columns=['ord_hour'], aggfunc=numpy.sum, fill_value=0)
    df_transpose_ord_hour.columns = ['ord_hour-' + str(col).zfill(2) for col in df_transpose_ord_hour.columns.values]
    df_pre_processed = df_pre_processed.join(df_transpose_ord_hour, how='inner')
    print(df_pre_processed.shape, df_pre_processed.head(10))

    # review_created_hour 기준 건수 집계
    df_review_created_hour = df_base[['shop_no', 'ord_dt', 'review_created_hour', 'ord_no']]
    df_group_review_created_hour = df_review_created_hour.groupby(['shop_no', 'ord_dt', 'review_created_hour']).agg(
        CNT_review_created_hour = ('ord_no', 'count'),
    )
    df_group_review_created_hour = df_group_review_created_hour.sort_index(ascending=True)
    df_group_review_created_hour = df_group_review_created_hour.reset_index()
    df_summary_review_created_hour = df_group_review_created_hour.groupby(['shop_no', 'ord_dt']).agg(
        MAX_review_created_hour = ('CNT_review_created_hour', max),
        MIN_review_created_hour = ('CNT_review_created_hour', min),
        SUM_review_created_hour = ('CNT_review_created_hour', sum),
        AVG_review_created_hour = ('CNT_review_created_hour', "mean"),
    )
    df_pre_processed = df_pre_processed.join(df_summary_review_created_hour, how='inner')
    print(df_pre_processed.shape, df_pre_processed.head(10))
    df_review_created_hour['constants'] = 1
    df_transpose_review_created_hour = pandas.pivot_table(df_review_created_hour, values='constants', index=['shop_no', 'ord_dt'], columns=['review_created_hour'], aggfunc=numpy.sum, fill_value=0)
    df_transpose_review_created_hour.columns = ['review_created_hour-' + str(col).zfill(2) for col in df_transpose_review_created_hour.columns.values]
    df_pre_processed = df_pre_processed.join(df_transpose_review_created_hour, how='inner')
    print(df_pre_processed.shape, df_pre_processed.head(10))
    # ===============================================================================================================================================================

    # ===============================================================================================================================================================
    # Target combine
    # ===============================================================================================================================================================
    df_target = pandas.read_csv(file_valid)
    df_target = df_target[['shop_no', 'ord_dt', 'abuse_yn']]
    df_summary_target = df_target.groupby(['shop_no', 'ord_dt']).agg(
        abuse_yn = ('abuse_yn', max),
    )
    df_pre_processed = df_pre_processed.join(df_summary_target, how='left')
    df_pre_processed = df_pre_processed.sort_index(ascending=True)
    print('abuse_yn', list(df_pre_processed['abuse_yn'].unique()))
    print(df_pre_processed.shape, df_pre_processed.head(10))
    # ===============================================================================================================================================================

    df_pre_processed.to_csv(file_pre_processed)


def main(p_args):
    # comma 데이터 수정
    # step_01_clean_csv()
    # ord_msg, item_name 파일 별도 분리
    # step_02_split_file()
    # Transpose 및 Target 결합
    # step_03_generate_base_code()

    df_check = pandas.read_csv(file_pre_processed)
    df_check = df_check[['shop_no', 'ord_dt', 'abuse_yn']]
    df_check = df_check.set_index(['ord_dt'])
    df_check = df_check.sort_index(ascending=True)
    df_check = df_check.reset_index()
    list_ord_dt = list(df_check['ord_dt'].unique())
    for ord_dt in list_ord_dt:
        df = df_check[df_check['ord_dt'] == ord_dt]
        print(ord_dt, ':', len(df['shop_no'].unique()))

    '''
    df_check['constants'] = 1
    df_check_T = pandas.pivot_table(df_check, values='constants', index=['shop_no', 'ord_dt'], columns=['abuse_yn'], aggfunc=numpy.sum, fill_value=0)
    df_check_T.columns = ['abuse_yn-' + str(col) for col in df_check_T.columns.values]
    df_check_T.to_csv('test.csv')
    '''

    pass



if __name__ == "__main__":
    # "error", "ignore", "always", "default", "module" or "once"
    warnings.filterwarnings('ignore')
    main(sys.argv)

