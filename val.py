#!/usr/bin/env python3
"""
YOLOv5 K-fold 모델 검증 스크립트
"""
import os
import argparse
import subprocess
from pathlib import Path
import pandas as pd
import yaml

def find_fold_models(runs_dir="runs/train", pattern="yolov5-rgbt-fold"):
    """
    K-fold 훈련된 모델들을 찾아서 반환
    """
    runs_path = Path(runs_dir)
    fold_models = []
    
    if not runs_path.exists():
        print(f"❌ {runs_dir} 디렉토리가 존재하지 않습니다.")
        return fold_models
    
    # 폴드별 모델 디렉토리 찾기
    for model_dir in runs_path.iterdir():
        if model_dir.is_dir() and pattern in model_dir.name:
            # best.pt 파일 확인
            best_model = model_dir / "weights" / "best.pt"
            last_model = model_dir / "weights" / "last.pt"
            
            if best_model.exists():
                fold_models.append({
                    'fold_name': model_dir.name,
                    'model_path': best_model,
                    'model_dir': model_dir,
                    'type': 'best'
                })
                print(f"✅ 발견: {model_dir.name} -> {best_model}")
            elif last_model.exists():
                fold_models.append({
                    'fold_name': model_dir.name,
                    'model_path': last_model,
                    'model_dir': model_dir,
                    'type': 'last'
                })
                print(f"✅ 발견: {model_dir.name} -> {last_model} (best.pt 없음)")
    
    return sorted(fold_models, key=lambda x: x['fold_name'])

def get_corresponding_yaml(fold_name, data_dir="data"):
    """
    폴드 이름에 맞는 YAML 파일 찾기
    """
    data_path = Path(data_dir)
    
    # fold 번호 추출
    if 'fold' in fold_name.lower():
        # yolov5-rgbt-fold1 -> fold1 추출
        fold_num = None
        for part in fold_name.split('-'):
            if 'fold' in part.lower():
                fold_num = part.lower().replace('fold', '')
                break
        
        if fold_num:
            yaml_files = [
                data_path / f"kaist-rgbt-fold{fold_num}.yaml",
                data_path / f"kaist-fold{fold_num}.yaml",
                data_path / f"fold{fold_num}.yaml"
            ]
            
            for yaml_file in yaml_files:
                if yaml_file.exists():
                    return yaml_file
    
    return None

def run_validation(model_path, data_yaml, output_dir, args):
    """
    단일 모델에 대해 검증 실행
    """
    cmd = [
        "python", "val.py",
        "--weights", str(model_path),
        "--data", str(data_yaml),
        "--img", str(args.imgsz),
        "--batch-size", str(args.batch_size),
        "--conf-thres", str(args.conf_thres),
        "--iou-thres", str(args.iou_thres),
        "--device", str(args.device),
        "--project", str(output_dir),
        "--name", Path(model_path).parent.parent.name + "_val",
        "--save-json",
        "--exist-ok"
    ]
    
    if args.save_txt:
        cmd.append("--save-txt")
    if args.verbose:
        cmd.append("--verbose")
    
    print(f"🚀 검증 명령어: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print("✅ 검증 완료")
            return True, result.stdout
        else:
            print(f"❌ 검증 실패: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print("❌ 검증 시간 초과 (1시간)")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ 검증 중 오류: {e}")
        return False, str(e)

def extract_metrics_from_output(output_text):
    """
    검증 결과에서 mAP 등 메트릭 추출
    """
    metrics = {
        'mAP50': 0.0,
        'mAP50-95': 0.0,
        'Precision': 0.0,
        'Recall': 0.0
    }
    
    lines = output_text.split('\n')
    for line in lines:
        if 'all' in line and len(line.split()) >= 7:
            try:
                parts = line.split()
                # YOLOv5 출력 형식: Class Images Instances P R mAP50 mAP50-95
                if len(parts) >= 7:
                    metrics['Precision'] = float(parts[4])
                    metrics['Recall'] = float(parts[5])
                    metrics['mAP50'] = float(parts[6])
                    metrics['mAP50-95'] = float(parts[7])
                break
            except (ValueError, IndexError):
                continue
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="YOLOv5 K-fold 모델 검증")
    parser.add_argument("--runs-dir", default="runs/train", help="훈련 결과 디렉토리")
    parser.add_argument("--data-dir", default="data", help="데이터셋 YAML 파일 디렉토리")
    parser.add_argument("--output-dir", default="runs/val", help="검증 결과 저장 디렉토리")
    parser.add_argument("--pattern", default="yolov5n-rgbt-fold", help="폴드 모델 디렉토리 패턴")
    
    # 검증 파라미터
    parser.add_argument("--imgsz", type=int, default=640, help="이미지 크기")
    parser.add_argument("--batch-size", type=int, default=32, help="배치 크기")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="IoU threshold")
    parser.add_argument("--device", default="0", help="GPU 디바이스")
    parser.add_argument("--save-txt", action="store_true", help="결과를 txt로 저장")
    parser.add_argument("--verbose", action="store_true", help="상세 출력")
    
    args = parser.parse_args()
    
    print("🔍 K-fold 모델 검증을 시작합니다...")
    print(f"   - 훈련 결과 디렉토리: {args.runs_dir}")
    print(f"   - 데이터 디렉토리: {args.data_dir}")
    print(f"   - 패턴: {args.pattern}")
    
    # 폴드 모델들 찾기
    fold_models = find_fold_models(args.runs_dir, args.pattern)
    
    if not fold_models:
        print("❌ 검증할 폴드 모델을 찾을 수 없습니다.")
        return
    
    print(f"\n📋 총 {len(fold_models)}개의 폴드 모델을 발견했습니다.")
    
    # 각 폴드별 검증 실행
    results = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, model_info in enumerate(fold_models, 1):
        print(f"\n{'='*60}")
        print(f"🔄 [{i}/{len(fold_models)}] {model_info['fold_name']} 검증 중...")
        print(f"   모델: {model_info['model_path']}")
        
        # 해당하는 YAML 파일 찾기
        data_yaml = get_corresponding_yaml(model_info['fold_name'], args.data_dir)
        
        if not data_yaml:
            print(f"❌ {model_info['fold_name']}에 해당하는 YAML 파일을 찾을 수 없습니다.")
            continue
        
        print(f"   데이터: {data_yaml}")
        
        # 검증 실행
        success, output = run_validation(
            model_info['model_path'], 
            data_yaml, 
            output_dir, 
            args
        )
        
        # 결과 저장
        result = {
            'Fold': model_info['fold_name'],
            'Model_Path': str(model_info['model_path']),
            'Data_YAML': str(data_yaml),
            'Success': success
        }
        
        if success:
            metrics = extract_metrics_from_output(output)
            result.update(metrics)
        else:
            result.update({
                'mAP50': 0.0,
                'mAP50-95': 0.0,
                'Precision': 0.0,
                'Recall': 0.0,
                'Error': output
            })
        
        results.append(result)
        
        print(f"   결과: {'✅ 성공' if success else '❌ 실패'}")
        if success:
            print(f"   - mAP@0.5: {result['mAP50']:.3f}")
            print(f"   - mAP@0.5:0.95: {result['mAP50-95']:.3f}")
            print(f"   - Precision: {result['Precision']:.3f}")
            print(f"   - Recall: {result['Recall']:.3f}")
    
    # 전체 결과 요약
    print(f"\n{'='*60}")
    print("📊 K-fold 검증 결과 요약")
    print(f"{'='*60}")
    
    # 성공한 결과들만 필터링
    successful_results = [r for r in results if r['Success']]
    
    if successful_results:
        df = pd.DataFrame(successful_results)
        
        # 평균 계산
        avg_map50 = df['mAP50'].mean()
        avg_map50_95 = df['mAP50-95'].mean()
        avg_precision = df['Precision'].mean()
        avg_recall = df['Recall'].mean()
        
        std_map50 = df['mAP50'].std()
        std_map50_95 = df['mAP50-95'].std()
        
        print(f"성공한 폴드: {len(successful_results)}/{len(results)}")
        print(f"\n평균 성능:")
        print(f"  - mAP@0.5: {avg_map50:.3f} ± {std_map50:.3f}")
        print(f"  - mAP@0.5:0.95: {avg_map50_95:.3f} ± {std_map50_95:.3f}")
        print(f"  - Precision: {avg_precision:.3f}")
        print(f"  - Recall: {avg_recall:.3f}")
        
        # 개별 결과 출력
        print(f"\n폴드별 상세 결과:")
        for result in successful_results:
            print(f"  {result['Fold']}: mAP@0.5={result['mAP50']:.3f}, mAP@0.5:0.95={result['mAP50-95']:.3f}")
        
        # CSV로 저장
        csv_path = output_dir / "kfold_validation_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n📄 상세 결과가 저장되었습니다: {csv_path}")
        
    else:
        print("❌ 성공한 검증 결과가 없습니다.")
    
    print(f"\n🎉 K-fold 검증이 완료되었습니다!")

if __name__ == "__main__":
    main()