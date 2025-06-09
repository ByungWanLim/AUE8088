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
    5-fold 교차 검증을 위한 YAML 파일 생성 (RGB-T 경로 문제 해결)
    
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
    print("원본 YAML 파일 내용:")
    print(yaml_data)
    
    # 경로 정보 추출
    dataset_path = yaml_data.get('path', 'datasets/kaist-rgbt')
    
    # train_file 처리 (리스트 또는 문자열)
    train_file_value = yaml_data.get('train', 'train-all-04.txt')
    
    # train_file이 리스트인 경우
    if isinstance(train_file_value, list):
        print(f"train 필드가 리스트입니다: {train_file_value}")
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
        
        # ✅ YAML 파일 생성 부분 수정 (클래스 불일치 문제 해결)
        # 새로운 YAML 데이터 생성 (기존 데이터 기반이지만 중요한 필드들은 올바르게 설정)
        yaml_content = {}
        
        # 경로에서 {} 플레이스홀더 제거
        clean_dataset_path = str(dataset_path).replace('{}/', '').replace('{}', '')
        yaml_content['path'] = clean_dataset_path
        
        # 절대 경로 사용
        current_dir = os.path.abspath(os.path.dirname(__file__))
        yaml_content['train'] = os.path.join(current_dir, 'data', 'folds', f'fold{fold_idx+1}', 'train.txt')
        yaml_content['val'] = os.path.join(current_dir, 'data', 'folds', f'fold{fold_idx+1}', 'val.txt')
        
        # ✅ KAIST RGB-T 데이터셋 전용 설정 (클래스 불일치 문제 해결)
        yaml_content['nc'] = 1  # KAIST RGB-T는 person detection이므로 클래스 1개
        yaml_content['names'] = ['person']  # 클래스 이름도 1개만
        
        # ✅ RGB-T 관련 추가 설정
        yaml_content['rgbt'] = True  # RGB-T 데이터셋임을 명시
        yaml_content['input_channels'] = 6  # RGB(3) + Thermal(3) = 6채널
        
        # ✅ 기존 YAML에서 유용한 정보가 있다면 복사 (문제가 되지 않는 필드들만)
        safe_fields_to_copy = ['description', 'authors', 'version', 'license', 'url']
        for field in safe_fields_to_copy:
            if field in yaml_data:
                yaml_content[field] = yaml_data[field]
        
        # YAML 파일 저장
        output_yaml_path = output_dir / f'kaist-rgbt-fold{fold_idx+1}.yaml'
        with open(output_yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
            
        print(f'✅ Fold {fold_idx+1} YAML 생성 완료: {output_yaml_path}')
        
        # ✅ 생성된 YAML 파일 내용 검증
        print(f"   - 훈련 이미지: {len(train_files_fold)}개")
        print(f"   - 검증 이미지: {len(val_files)}개")
        print(f"   - 데이터셋 경로: {clean_dataset_path}")
        print(f"   - 클래스 수: {yaml_content['nc']}")
        print(f"   - 클래스 이름: {yaml_content['names']}")
    
    print(f'🎉 모든 {n_folds}-fold 교차 검증 파일 생성 완료!')
    
    # ✅ 추가: 생성된 YAML 파일들 목록 출력
    print("\n📋 생성된 YAML 파일들:")
    for i in range(1, n_folds + 1):
        yaml_path = output_dir / f'kaist-rgbt-fold{i}.yaml'
        if yaml_path.exists():
            print(f"   - {yaml_path}")

def validate_yaml_files(output_dir='data', n_folds=5):
    """
    생성된 YAML 파일들의 유효성 검사
    """
    print("\n🔍 YAML 파일 유효성 검사...")
    
    for i in range(1, n_folds + 1):
        yaml_path = Path(output_dir) / f'kaist-rgbt-fold{i}.yaml'
        
        if not yaml_path.exists():
            print(f"❌ {yaml_path} 파일이 존재하지 않습니다.")
            continue
        
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # 필수 필드 확인
            required_fields = ['path', 'train', 'val', 'nc', 'names']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                print(f"❌ {yaml_path}: 누락된 필드 - {missing_fields}")
                continue
            
            # ✅ 클래스 일치성 확인
            nc = data['nc']
            names = data['names']
            if len(names) != nc:
                print(f"❌ {yaml_path}: 클래스 불일치 - nc={nc}, names 길이={len(names)}")
                continue
            
            # 파일 경로 존재 확인
            train_file_exists = os.path.exists(data['train'])
            val_file_exists = os.path.exists(data['val'])
            
            if not train_file_exists or not val_file_exists:
                print(f"⚠️  {yaml_path}: 파일 경로 문제")
                print(f"   - 훈련 파일 존재: {train_file_exists}")
                print(f"   - 검증 파일 존재: {val_file_exists}")
            else:
                print(f"✅ {yaml_path} - 모든 검사 통과")
                print(f"   - 클래스 수: {nc}")
                print(f"   - 클래스 이름: {names}")
                print(f"   - RGB-T 설정: {data.get('rgbt', False)}")
                
                # 실제 이미지 수 확인
                with open(data['train'], 'r') as f:
                    train_count = len([line.strip() for line in f.readlines() if line.strip()])
                with open(data['val'], 'r') as f:
                    val_count = len([line.strip() for line in f.readlines() if line.strip()])
                
                print(f"   - 훈련 이미지: {train_count}개")
                print(f"   - 검증 이미지: {val_count}개")
                
        except Exception as e:
            print(f"❌ {yaml_path} 읽기 오류: {e}")

if __name__ == "__main__":
    # K-fold YAML 파일 생성
    create_kfold_yaml()
    
    # 생성된 파일들 검증
    validate_yaml_files()