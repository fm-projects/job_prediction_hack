import pandas as pd
import json
from tqdm import tqdm
from scipy import stats
import numpy as np
from catboost import CatBoostRegressor
import copy


def prepare_df(df, is_test=True):
    df = df.drop(columns=((df.isnull().sum().sort_values() > df.shape[0] * 0.9)[-18:]).index.values.tolist()) # дропаем колонки, в которых больше 90%: "NaN"
    df = df.drop(columns=['languageKnowledge', 'hardSkills', 'softSkills']) # дропаем колонки, в которых больше 90%: "[]"
    df = df.drop(columns=['deleted', 'is_moderated', 'visibility']) # дропаем колонки, в записях которых одни и те же значения (is_moderated: True, deleted: False)
    df = df.drop(columns=['company']) # дропаем колонку, так как информация из этой колонки есть в других
    df = df.drop(columns=['salary']) # дропаем колонку, так как она совпадает с колонкой "min_salary"
    df = df.drop(columns=['id', 'data_ids'])
    df = df.drop(columns=['publication_period', 'bonus_type', 'is_uzbekistan_recruitment', 'status', 'source_type']) # дропаем колонки, значение feature importance которых меньше 

    df = df.drop(columns=['salary_max'])
    
    df['change_time'] = pd.to_datetime(df['change_time'])
    df['date_create'] = pd.to_datetime(df['date_create'])
    df['date_modify'] = pd.to_datetime(df['date_modify'])
    df['published_date'] = pd.to_datetime(df['published_date'])
    
    for i in ['date_create', 'date_modify', 'published_date', 'change_time']:
        df[i] = pd.to_datetime(df[i])
        df[f'{i}_year'] = df[i].dt.year
        df[f'{i}_month'] = df[i].dt.month
        df[f'{i}_day'] = df[i].dt.month
        df[f'{i}_day_of_week'] = df[i].dt.dayofweek
    
    df = df.drop(columns=['date_create', 'date_modify', 'published_date', 'change_time', 'company_code', 'vacancy_address_house'])
    
    if not is_test:
        z = np.abs(stats.zscore(df['salary_min']))
        df = df[(z < 3)]

    return df

def fill_na(df, features):
    for feature in features:
        df[feature].fillna('', inplace=True)
        
text_features = ['additional_requirements', 'contact_person', 'education',
                 'education_speciality', 'other_vacancy_benefit',
                 'position_requirements','position_responsibilities',
                 'regionName', 'vacancy_address_additional_info',
                 'vacancy_address', 'full_company_name', 'professionalSphereName',
                 'vacancy_name']

cat_features = ['busy_type', 'code_external_system',
                'code_professional_sphere', 'original_source_type',
                'company_business_size', 'retraining_grant',
                'required_drive_license', 'schedule_type',
                'measure_type']

    
if __name__ == '__main__':
    df = pd.read_csv('TRAIN_SAL.csv')
    prepared_df = prepare_df(copy.deepcopy(df))
    fill_na(prepared_df, text_features)
    fill_na(prepared_df, cat_features)
    
    model = CatBoostRegressor()
    model.load_model('SAl_model')
    prepared_df = prepared_df.drop(columns=['salary_min'])
    predicted = model.predict(prepared_df)
    SAL_part = pd.DataFrame([])
    SAL_part['id'] = df['id']
    SAL_part['salary'] = predicted
    SAL_part['task_type'] = 'SAL'
    SAL_part.to_csv('sub.csv')
