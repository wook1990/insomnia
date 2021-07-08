# -*- coding: utf-8 -*-
import sys
import pandas
import numpy
import warnings
import datetime

DICT_MIN = dict()
DICT_MAX = dict()
#df_min_max = pandas.read_csv('/opt/wai/workspaces/kogas/data/KOGAS_FB_min_max_values_02_derive.csv')
df_min_max = pandas.read_csv('G:/내 드라이브/WAI_WORK/Project_I/2020/03. 가스공사/kogas/data/KOGAS_FB_min_max_values_02_derive.csv')
for _, row in df_min_max.iterrows():
    DICT_MIN[row['FEATURE'].replace('LOG-MOV-', 'LOG-').replace('ROT-MOV-', 'ROT-')] = row['MIN']
    DICT_MAX[row['FEATURE'].replace('LOG-MOV-', 'LOG-').replace('ROT-MOV-', 'ROT-')] = row['MAX']


def build_data(p_source_file_name, p_target_file_name):

    COUNT_FOR_WRITING = 2000
    NULL_VALUE = 3.141592653589793

    set_for_checking_init_column = set(['BAS-start_temp','BAS-fan_rpm','BAS-start_hum','BAS-group_vapor_flow','BAS-VAR_CLASS','BAS-unit_lng_temp','BAS-working_hour_gr','BAS-working_time'])
    set_for_checking_init_column.add('raw_id')
    set_for_checking_init_column.add('target')
    set_for_checking_mov_column = set(['BAS-unit_lng_temp','BAS-working_hour_gr','BAS-working_time','BAS-group_vapor_flow'])
    set_for_checking_sqr_column = set(['BAS-start_temp','MOV-BAS-working_time','BAS-start_hum','BAS-group_vapor_flow','BAS-VAR_CLASS','MOV-BAS-working_hour_gr'])
    set_for_checking_rot_column = set(['BAS-working_time','BAS-fan_rpm','BAS-start_hum','BAS-group_vapor_flow'])
    set_for_checking_log_column = set(['BAS-unit_lng_temp','BAS-working_hour_gr','BAS-VAR_CLASS'])
    set_for_checking_grp_column = set(['BAS-start_temp','BAS-fan_rpm','LOG-BAS-VAR_CLASS','SQR-BAS-start_hum','ROT-BAS-start_hum','SQR-BAS-group_vapor_flow','ROT-MOV-BAS-group_vapor_flow','SQR-BAS-VAR_CLASS','LOG-MOV-BAS-unit_lng_temp','LOG-MOV-BAS-working_hour_gr','ROT-MOV-BAS-working_time','MOV-BAS-unit_lng_temp','SQR-MOV-BAS-working_hour_gr','SQR-BAS-start_temp','ROT-BAS-fan_rpm','SQR-MOV-BAS-working_time'])
    set_for_checking_continuous_column = set([''])

    list_col_for_saving = ['raw_id','target','BAS-start_temp','BAS-fan_rpm','BAS-start_hum','BAS-group_vapor_flow','BAS-VAR_CLASS','BAS-unit_lng_temp','BAS-working_hour_gr','BAS-working_time']

    file_source = open(p_source_file_name, 'r')
    list_to_write = list()
    list_sorted_columns = list()

    list_columns = str(file_source.readline()).strip().split(',')
    flag_while_loop = True
    loop_count = 0
    while flag_while_loop:
        line = str(file_source.readline()).strip()
        if not line:
            flag_while_loop = False
            continue
        list_values = line.split(',')
        dict_cell = dict()
        for values in zip(list_columns, list_values):
            key = values[0]
            if values[1] == '':
                value = NULL_VALUE
            else:
                value = values[1]

            if 'raw_id' in key:
                dict_cell[key] = str(value)
            elif 'target' in key:
                dict_cell[key] = int(float(value))
            elif 'BAS-' + key in set_for_checking_init_column:
                dict_cell['BAS-' + key] = float(value)

        list_to_write.append(dict_cell)
        loop_count += 1

        if loop_count % COUNT_FOR_WRITING == 0:
            df = pandas.DataFrame(list_to_write)

            for column_name in list(set_for_checking_mov_column):
                if column_name == '':
                    break
                df['MOV-' + column_name] = df[column_name] - DICT_MIN[column_name] + 1

            for column_name in list(set_for_checking_sqr_column):
                if column_name == '':
                    break
                df['SQR-' + column_name] = df[column_name] * df[column_name]

            for column_name in list(set_for_checking_rot_column):
                if column_name == '':
                    break
                if DICT_MIN[column_name] < 0.0:
                    df['ROT-' + column_name] = numpy.sqrt(df[column_name] - DICT_MIN[column_name] + 1)
                else:
                    df['ROT-' + column_name] = numpy.sqrt(df[column_name])

            for column_name in list(set_for_checking_log_column):
                if column_name == '':
                    break
                if DICT_MIN[column_name] < 0.0:
                    df['LOG-' + column_name] = numpy.log(df[column_name] - DICT_MIN[column_name] + 1)
                else:
                    df['LOG-' + column_name] = numpy.log(df[column_name])

            column_names = df.columns
            for column_name in column_names:
                if 'raw_id' in column_name:
                    continue
                if 'target' in column_name:
                    continue
                range_value = (DICT_MAX[column_name] - DICT_MIN[column_name])
                if float(range_value) == 0.0:
                    range_value = 1.0
                if column_name in set_for_checking_continuous_column:
                    df[column_name] = numpy.log((df[column_name] - DICT_MIN[column_name]) / range_value + 1)
                else:
                    df[column_name] = (df[column_name] - DICT_MIN[column_name]) / range_value
                group_column_name = column_name
                if group_column_name in set_for_checking_grp_column:
                    df['GRP-' + group_column_name] = df[column_name] * 20
                    df = df.replace([numpy.inf, -numpy.inf], numpy.nan)
                    df = df.fillna(-1)
                    df['GRP-' + group_column_name] = df['GRP-' + group_column_name].astype(float)
                    df['GRP-' + group_column_name] = df['GRP-' + group_column_name].astype(int)

            df = df[list_col_for_saving]
            if loop_count == COUNT_FOR_WRITING:
                df.to_csv(p_target_file_name, index=None)
            else:
                df.to_csv(p_target_file_name, index=None, mode='a', header=None)
            list_to_write = list()
            print(datetime.datetime.today().strftime('[%Y/%m/%d %H:%M:%S]'), str(loop_count).zfill(10), flush=True)

    if len(list_to_write) > 0:
        df = pandas.DataFrame(list_to_write)

        for column_name in list(set_for_checking_mov_column):
            if column_name == '':
                break
            df['MOV-' + column_name] = df[column_name] - DICT_MIN[column_name] + 1

        for column_name in list(set_for_checking_sqr_column):
            if column_name == '':
                break
            df['SQR-' + column_name] = df[column_name] * df[column_name]

        for column_name in list(set_for_checking_rot_column):
            if column_name == '':
                break
            if DICT_MIN[column_name] < 0.0:
                df['ROT-' + column_name] = numpy.sqrt(df[column_name] - DICT_MIN[column_name] + 1)
            else:
                df['ROT-' + column_name] = numpy.sqrt(df[column_name])

        for column_name in list(set_for_checking_log_column):
            if column_name == '':
                break
            if DICT_MIN[column_name] < 0.0:
                df['LOG-' + column_name] = numpy.log(df[column_name] - DICT_MIN[column_name] + 1)
            else:
                df['LOG-' + column_name] = numpy.log(df[column_name])

        column_names = df.columns
        for column_name in column_names:
            if 'raw_id' in column_name:
                continue
            if 'target' in column_name:
                continue
            range_value = (DICT_MAX[column_name] - DICT_MIN[column_name])
            if float(range_value) == 0.0:
                range_value = 1.0
            if column_name in set_for_checking_continuous_column:
                df[column_name] = numpy.log((df[column_name] - DICT_MIN[column_name]) / range_value + 1)
            else:
                df[column_name] = (df[column_name] - DICT_MIN[column_name]) / range_value
            group_column_name = column_name
            if group_column_name in set_for_checking_grp_column:
                df['GRP-' + group_column_name] = df[column_name] * 20
                df = df.replace([numpy.inf, -numpy.inf], numpy.nan)
                df = df.fillna(-1)
                df['GRP-' + group_column_name] = df['GRP-' + group_column_name].astype(float)
                df['GRP-' + group_column_name] = df['GRP-' + group_column_name].astype(int)

        df = df[list_col_for_saving]
        if loop_count < COUNT_FOR_WRITING:
            df.to_csv(p_target_file_name, index=None)
        else:
            df.to_csv(p_target_file_name, mode='a', index=None, header=None)
    print(datetime.datetime.today().strftime('[%Y/%m/%d %H:%M:%S]'), str(loop_count).zfill(10), flush=True)

    file_source.close()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    #source_file_name = sys.argv[1]
    #target_file_name = sys.argv[2]
    '''
    # train_set
    source_file_name = "G:/내 드라이브/WAI_WORK/Project_I/2020/03. 가스공사/kogas/data/kogas_train.csv"
    target_file_name = "C:/Users/WAI/Documents/kogas/data/kogas_train_model.csv"

    print(datetime.datetime.today().strftime('[%Y/%m/%d %H:%M:%S]'), 'START', flush=True)
    build_data(source_file_name, target_file_name)
    print(datetime.datetime.today().strftime('[%Y/%m/%d %H:%M:%S]'), 'FINISH', flush=True)
    '''

    # test set
    source_file_name = "G:/내 드라이브/WAI_WORK/Project_I/2020/03. 가스공사/kogas/data/opt_data.csv"
    target_file_name = "C:/Users/WAI/Documents/kogas/data/kogas_opt_model.csv"

    print(datetime.datetime.today().strftime('[%Y/%m/%d %H:%M:%S]'), 'START', flush=True)
    build_data(source_file_name, target_file_name)
    print(datetime.datetime.today().strftime('[%Y/%m/%d %H:%M:%S]'), 'FINISH', flush=True)
