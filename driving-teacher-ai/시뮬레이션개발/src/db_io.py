from sqlalchemy import create_engine
import pandas as pd
import geopandas as gpd
class Engine():
    def __init__(self, db_info):
        '''
        db_info(필) : user_name, password, host, port, db_name 이 key로 포함된 dict 형식
        '''
        self.engine = create_engine("postgresql://{user_name}:{password}@{host}:{port}/{db_name}".format(**db_info))


    def select(self, table:str, schema:str='', columns:list = [], wheres:list = [], orderbys:list = [], gis_col:str = ''):
        '''
        table: 테이블명
        schema: 스키마명
        columns(list): 컬럼명(str)이 원소인 리스트
        wheres(list): where구문(str)이 원소인 리스트
        '''
        sql = ''
        str_select = 'select '
        if columns is None:
            str_select += '*'
        else:
            str_select += '{}'.format(', '.join([f'"{x}"' for x in columns]))
            
        str_from = ' from '
        if schema != '':
            str_from = str_from + schema + '.'
        str_from += table

        str_where = ''
        if len(wheres) > 0:
            str_where += ' where '
            str_where += '{}'.format(' and '.join(wheres))
        str_orderby = ''
        if len(orderbys) > 0:
            str_orderby += ' order by '
            str_orderby += '{}'.format(' and '.join(orderbys))

        sql = str_select + str_from + str_where + str_orderby
        if gis_col != '':
            return gpd.read_postgis(sql = sql, con = self.engine, index_col = None, geom_col = gis_col) 
        else:
            return pd.read_sql(sql = sql, con = self.engine, index_col = None)







