import os
import random
import numpy as np
from pathlib import Path
import yaml

def create_kfold_yaml(
    yaml_file='data/kaist-rgbt.yaml',
    output_dir='data',
    n_folds=5,
    seed=42
):
    """
    5-fold 교차 검증을 위한 YAML 파일 생성
    
    Args:
        yaml_file: 기존 YAML 파일 경로
        output_dir: 출력 디렉토리
        n_folds: 폴드 수
        seed: 랜덤 시드
    """
    # 출력 디렉토리 생성
    output_dir = Path(output_dir)
    fold_dir = output_dir / 'folds'
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # 기존 YAML 파일 읽기
    with open(yaml_file, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    # YAML 파일 내용 출력 (디버깅용)
    print("YAML 파일 내용:")
    print(yaml_data)
    
    # 경로 정보 추출
    dataset_path = yaml_data.get('path', 'datasets/kaist-rgbt')
    
    # train_file 처리 (리스트 또는 문자열)
    train_file_value = yaml_data.get('train', 'train-all-04.txt')
    
    # train_file이 리스트인 경우
    if isinstance(train_file_value, list):
        print(f"train 필드가 리스트입니다: {train_file_value}")
        # 첫 번째 항목만 사용하거나, 모든 파일을 합쳐서 사용할 수 있음
        # 여기서는 첫 번째 항목만 사용
        train_file = train_file_value[0] if train_file_value else 'train-all-04.txt'
    else:
        train_file = train_file_value
    
    # test_file 처리 (리스트 또는 문자열)
    test_file_value = yaml_data.get('test', 'test-all-20.txt')
    
    if isinstance(test_file_value, list):
        print(f"test 필드가 리스트입니다: {test_file_value}")
        test_file = test_file_value[0] if test_file_value else 'test-all-20.txt'
    else:
        test_file = test_file_value
    
    # 상대 경로를 절대 경로로 변환
    if isinstance(train_file, str) and not os.path.isabs(train_file):
        if train_file.startswith('../'):
            train_file = train_file[3:]  # '../' 제거
        else:
            train_file = os.path.join(dataset_path, train_file)
            
    if isinstance(test_file, str) and not os.path.isabs(test_file):
        if test_file.startswith('../'):
            test_file = test_file[3:]  # '../' 제거
        else:
            test_file = os.path.join(dataset_path, test_file)
    
    print(f"학습 데이터 파일: {train_file}")
    print(f"테스트 데이터 파일: {test_file}")
    
    # 확장자 확인 및 수정 (train-all-04.test가 오타인 경우)
    if isinstance(train_file, str):
        if not os.path.exists(train_file) and train_file.endswith('.txt'):
            test_file_path = train_file.replace('.txt', '.test')
            if os.path.exists(test_file_path):
                print(f"경고: {train_file}가 존재하지 않지만 {test_file_path}가 존재합니다. 이 파일을 사용합니다.")
                train_file = test_file_path
    
    # 파일 존재 확인
    if isinstance(train_file, str) and not os.path.exists(train_file):
        raise FileNotFoundError(f"학습 데이터 파일을 찾을 수 없습니다: {train_file}")
    
    # 전체 학습 데이터 파일 목록 로드
    with open(train_file, 'r') as f:
        train_files = [line.strip() for line in f.readlines()]
    
    print(f"총 {len(train_files)}개의 학습 이미지를 {n_folds}개의 폴드로 나눕니다.")
    
    # 무작위로 섞기
    random.seed(seed)
    random.shuffle(train_files)
    
    # n_folds 개의 폴드로 나누기
    fold_size = len(train_files) // n_folds
    folds = []
    for i in range(n_folds):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < n_folds - 1 else len(train_files)
        folds.append(train_files[start_idx:end_idx])
    
    # 각 폴드별로 train/val 파일 생성 및 YAML 파일 생성
    for fold_idx in range(n_folds):
        # 현재 폴드를 검증 세트로, 나머지를 학습 세트로 사용
        val_files = folds[fold_idx]
        train_files_fold = []
        for i in range(n_folds):
            if i != fold_idx:
                train_files_fold.extend(folds[i])
        
        # train.txt와 val.txt 파일 생성
        fold_path = fold_dir / f'fold{fold_idx+1}'
        fold_path.mkdir(exist_ok=True)
        
        with open(fold_path / 'train.txt', 'w') as f:
            f.write('\n'.join(train_files_fold))
        
        with open(fold_path / 'val.txt', 'w') as f:
            f.write('\n'.join(val_files))
        
        # YAML 파일 생성 부분 수정
        yaml_content = yaml_data.copy()
        yaml_content['path'] = dataset_path
        # 절대 경로 사용
        current_dir = os.path.abspath(os.path.dirname(__file__))
        yaml_content['train'] = os.path.join(current_dir, 'data', 'folds', f'fold{fold_idx+1}', 'train.txt')
        yaml_content['val'] = os.path.join(current_dir, 'data', 'folds', f'fold{fold_idx+1}', 'val.txt')
        # 하드코딩된 Ubuntu 경로 제거
        if 'download' in yaml_content:
            yaml_content.pop('download')
        
        with open(output_dir / f'kaist-rgbt-fold{fold_idx+1}.yaml', 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
            
        print(f'Fold {fold_idx+1} YAML 생성 완료: {output_dir}/kaist-rgbt-fold{fold_idx+1}.yaml')
    
    print(f'모든 {n_folds}-fold 교차 검증 파일 생성 완료!')

if __name__ == "__main__":
    create_kfold_yaml()