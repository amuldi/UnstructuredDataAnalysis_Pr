# ===== 1) 라이브러리 불러오기 =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os,re,glob
from scipy.stats import skew

# 시각화 기본 테마/해상도 설정
sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 100
print('✅ 라이브러리 임포트 완료')

# ===== 2) 데이터 경로 설정 및 파일 탐색 =====
data_dir = '/Users/jsh/Desktop/class/3-1/비정형데이터분석/실습/A_DeviceMotion_data 복사본'
os.chdir(data_dir)

# 하위 폴더까지 포함하여 CSV 파일 전체 탐색
fls = glob.glob("**/sub_*.csv", recursive=True)

print(f'총 {len(fls)}개 파일 발견:')
print(fls[:10])  # 처음 10개만 미리 확인

# ===== 3) 모든 CSV를 딕셔너리로 로드 =====
# key: 파일 경로, value: 해당 파일 DataFrame
data_dict = {}
for f in fls:
    data_dict[f] = pd.read_csv(f)

print(f'✅ 총 {len(data_dict)}개 파일 로드 완료')

# 로드된 데이터 샘플 확인
first_key = list(data_dict.keys())[0]
print(f'\n샘플 데이터 ({first_key}):')
print(data_dict[first_key].head())

# ===== 4) Subject #1의 Walking 데이터만 추출 ====
user1 = [f for f in fls if f.endswith('sub_1.csv')]
print('✅ Subject #1 파일 목록:')
print(user1)

user_walking = [f for f in user1 if 'wlk' in f]
print('Subject #1 Walking 파일 목록:')
print(user_walking)

# ===== 5) 파일명에서 숫자 추출 예시 =====
# 예: wlk_exp8_sub19 -> ['8', '19']
example_f = 'wlk_exp8_sub19'
nums = re.findall(r'\d+', example_f)

print(f'파일명: {example_f}')
print(f'추출된 숫자 목록: {nums}')
print(f'exp_no = {nums[0]}, id = {nums[1]}')

# ===== 6) Subject #1 Walking 파일 병합 =====
frames = []
for f in user_walking:
    temp = data_dict[f].copy()
    nums = re.findall(r'\d+', f) # 파일명에서 숫자 모두 추출
    temp['exp_no'] = nums[0] # 첫 번째 숫자 = exp_no
    temp['id'] = nums[1] # 두 번째 숫자 = id
    frames.append(temp)

# 실험 파일들을 하나의 DataFrame으로 결합
user1_walking_total = pd.concat(frames, ignore_index=True)

print(f'✅ 통합 DataFrame 크기: {user1_walking_total.shape}')
user1_walking_total.head()

# ===== 7) 벡터 크기(magnitude) 계산 함수 =====
# var.x, var.y, var.z 컬럼으로부터 sqrt(x^2 + y^2 + z^2) 계산
def mag(df, var):
    df = df.copy()
    df[f'mag{var}'] = np.sqrt(
        df[f'{var}.x']**2 +
        df[f'{var}.y']**2 +
        df[f'{var}.z']**2
    )
    return df

# userAcceleration magnitude 생성
user1_walking_total = mag(user1_walking_total, 'userAcceleration')

print('✅ maguserAcceleration 컬럼 생성 완료')

# ===== 8) 실험(exp_no)별 상대 시간 인덱스 생성 =====
# 각 exp_no 그룹 내에서 1,2,3,... 순서 부여
user1_walking_total['time'] = (
    user1_walking_total
.groupby('exp_no')
.cumcount() + 1
)
print('✅ time 컬럼 생성 완료')
print(user1_walking_total.groupby('exp_no')['time'].max())

# ===== 9) Subject #1 Walking 시각화 =====
# exp_no별로 분할하여 magnitude 시계열 확인
g = sns.FacetGrid(
user1_walking_total,
col='exp_no',
col_wrap=3, # 3열씩 배치
height=3,
sharey=False
)
g.map(plt.plot, 'time', 'maguserAcceleration', linewidth=0.8)
g.set_axis_labels('Time (relative)', 'Acceleration Magnitude')
g.figure.suptitle('Subject #1 Walking — Acceleration Magnitude', 
y=1.02, fontsize=14)
plt.tight_layout()
plt.show()

# ===== 10) 전체 HAR 데이터 통합 =====
frames = []
for f in fls:
    temp = data_dict[f].copy()
    nums = re.findall(r'\d+', f)
    temp['exp_no'] = nums[0]
    temp['id'] = nums[1] 
    temp['activity'] = f.split('_')[0]  # 파일명 앞부분을 활동명으로 사용
    frames.append(temp)

HAR_total = pd.concat(frames, ignore_index=True)

print(f'✅ 전체 HAR DataFrame 크기: {HAR_total.shape}')
print(f'포함된 활동 종류: {HAR_total["activity"].unique()}')
HAR_total.head(100)

# ===== 11) 전체 데이터의 magnitude 변수 생성 =====
HAR_total=mag(HAR_total, 'userAcceleration')
HAR_total=mag(HAR_total, 'rotationRate')

print('✅ Magnitude 변수 생성 완료')
print(HAR_total.info())
HAR_total[['maguserAcceleration', 'magrotationRate']].describe()

HAR_total.to_pickle('HAR_total.pkl')
print('✅ HAR_total.pkl 저장 완료')

# ===== 12) 사용자/실험/활동 단위 요약 통계 =====
# mean, min, max, std, skew(왜도) 계산
HAR_summary = (
    HAR_total
.groupby(['id', 'exp_no', 'activity'])[
['maguserAcceleration', 'magrotationRate']
]
.agg(['mean', 'min', 'max', 'std', skew])
.reset_index()
)

# 다중 컬럼 인덱스를 단일 컬럼명으로 평탄화
HAR_summary.columns = [
    '_'.join(col).strip('_') if isinstance(col, tuple) else col
for col in HAR_summary.columns
]

print(f'✅ HAR_summary 크기: {HAR_summary.shape}')
HAR_summary.head(10)
file_name='HAR_summary.csv'
HAR_summary.to_csv(file_name, index=False)


# ===== 13) Subject #1 파일 다시 추출 =====
# user1 파일 추출
user1 = [f for f in fls if f.endswith('sub_1.csv')]
print('✅ Subject #1 파일 목록:')
print(user1)

# 🔥 jogging 데이터만 추출
user_jogging = [f for f in user1 if 'jog' in f]
print('Subject #1 Jogging 파일 목록:')
print(user_jogging)


# ===== 14) Subject #1 Jogging 데이터 병합 =====
# 데이터 병합
frames = []
for f in user_jogging:
    temp = data_dict[f].copy()
    nums = re.findall(r'\d+', f)
    temp['exp_no'] = nums[0]
    temp['id'] = nums[1]
    frames.append(temp)

user1_jogging_total = pd.concat(frames, ignore_index=True)

print(f'✅ 통합 DataFrame 크기: {user1_jogging_total.shape}')
user1_jogging_total.head()


# ===== 15) Jogging 데이터에 magnitude 적용 =====
# magnitude 함수 적용
user1_jogging_total = mag(user1_jogging_total, 'userAcceleration')

print('✅ maguserAcceleration 컬럼 생성 완료')


# ===== 16) Jogging 데이터 time 생성 =====
# time 생성
user1_jogging_total['time'] = (
    user1_jogging_total
    .groupby('exp_no')
    .cumcount() + 1
)

print('✅ time 컬럼 생성 완료')
print(user1_jogging_total.groupby('exp_no')['time'].max())


# ===== 17) Subject #1 Jogging 시각화 =====
# 🔥 시각화
g = sns.FacetGrid(
    user1_jogging_total,
    col='exp_no',
    col_wrap=3,
    height=3,
    sharey=False
)

g.map(plt.plot, 'time', 'maguserAcceleration', linewidth=0.8)

g.set_axis_labels('Time (relative)', 'Acceleration Magnitude')

g.figure.suptitle(
    'Subject #1 Jogging — Acceleration Magnitude',
    y=1.02,
    fontsize=14
)

plt.tight_layout()
plt.show()
