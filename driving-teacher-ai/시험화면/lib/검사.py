def 검증_unique(data, columns = []):
    data2 = data.drop_duplicates(subset = columns)
    print('unique 조건 위반 record 수: ', len(data) - len(data2))
    return data2

def 검증_notnull(data, columns = []):
    data2 = data.dropna(subset = columns)
    print('not null 조건 위반 record 수: ', len(data) - len(data2))
    return data2

def 레코드수(data, columns = []):
    data = 검증_unique(data, columns)
    data = 검증_notnull(data, columns)
    return len(data)

def 레코드수(data, columns = []):
    data = 검증_unique(data, columns)
    data = 검증_notnull(data, columns)
    return len(data)