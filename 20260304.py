import pandas as pd # 데이터 프레임, CSV 파일 읽기
import numpy as np # 벡터 연산, 수치 계산  
import os # 폴더 경로, 파일 탐색
import glob # 파일 목록 패턴 매칭

# 데이터셋 경로 설정
dataset_path = '/Users/jsh/Desktop/study/3-1/비정형데이터분석/motion-sense-master'

# 경로 확인
if os.path.exists(dataset_path):
    print(f'데이터 경로 확인:{dataset_path}')
else:
    print('데이터 경로가 존재하지 않습니다.')

# 전체 데이터 읽기
all_data = [] 

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(root, file)
            df_temp = pd.read_csv(file_path)
            df_temp['file_name'] = file
            all_data.append(df_temp)
final_df = pd.concat(all_data, ignore_index = True)

print(final_df.head()) # 데이터 상위 5행 확인
print(final_df.describe()) # 데이터 통계 요약

# python 함수 구현
def calculate_magnitude(df, x_col, y_col, z_col): # x,y,z 컬럼을 받아 벡터의 크기(magnitude)를 반환하는 함수
    x = df[x_col]
    y = df[y_col]
    z = df[z_col]


    magnitude = np.sqrt(x**2 + y**2 + z**2) # 제곱의 합의 제곱근 연산
    return magnitude

# 파생변수 생성
final_df['mag_acc'] = calculate_magnitude(
    final_df,'userAcceleration.x','userAcceleration.y','userAcceleration.z'
)

# 결과 확인
print(final_df[['userAcceleration.x', 'mag_acc']].head())

# 분석 상태 저장  
final_df.to_csv('processed_data.csv', index=False) # CSV 파일로 저장




