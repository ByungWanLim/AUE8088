#!/usr/bin/env python3
"""
YOLO11n RGB-T Training Script for KAIST dataset with W&B logging and Config file support
"""
import argparse
from ultralytics import YOLO
import yaml
import torch
import os
import wandb
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="model path")
    parser.add_argument("--cfg", type=str, default="", help="model config file path (*.yaml)")
    parser.add_argument("--data", type=str, required=True, help="dataset.yaml path")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="image size")
    parser.add_argument("--device", default="0", help="cuda device: 0, 1, 2, 3 or cpu")
    parser.add_argument("--project", default="runs/train", help="project name")
    parser.add_argument("--name", default="yolo11n_kaist", help="experiment name")
    parser.add_argument("--workers", type=int, default=8, help="dataloader workers")
    parser.add_argument("--patience", type=int, default=50, help="early stopping patience")
    parser.add_argument("--resume", type=str, default="", help="resume training from checkpoint")
    parser.add_argument("--save-period", type=int, default=10, help="save model every N epochs")
    
    # W&B arguments (train_simple.py 참고)
    parser.add_argument("--entity", default=None, help="W&B entity")
    parser.add_argument("--wandb-project", default="KAIST-RGBT-YOLO11n", help="W&B project name")
    parser.add_argument("--wandb-name", default=None, help="W&B run name")
    parser.add_argument("--no-wandb", action="store_true", help="disable W&B logging")
    
    args = parser.parse_args()
    
    # Config 파일 검증
    if args.cfg:
        if not os.path.exists(args.cfg):
            raise FileNotFoundError(f"❌ Config 파일을 찾을 수 없습니다: {args.cfg}")
        if not args.cfg.endswith(('.yaml', '.yml')):
            raise ValueError(f"❌ Config 파일은 YAML 형식이어야 합니다: {args.cfg}")
        print(f"📄 Config 파일 사용: {args.cfg}")
        
        # Config 파일 내용 읽기 및 검증
        try:
            with open(args.cfg, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            print(f"✅ Config 파일 로드 완료: {len(config_data) if config_data else 0}개 설정")
        except Exception as e:
            raise ValueError(f"❌ Config 파일 파싱 오류: {e}")
    
    # GPU 디바이스 설정 및 확인 (수정된 부분)
    gpu_name = "Unknown GPU"
    gpu_memory = 0
    
    if args.device != "cpu":
        if not torch.cuda.is_available():
            print("❌ CUDA를 사용할 수 없습니다. CPU로 전환합니다.")
            args.device = "cpu"
        else:
            try:
                gpu_id = int(args.device)
                if gpu_id >= torch.cuda.device_count():
                    print(f"❌ GPU {gpu_id}가 존재하지 않습니다. 사용 가능한 GPU: 0-{torch.cuda.device_count()-1}")
                    gpu_id = 0
                    args.device = "0"
                
                # GPU 정보를 먼저 가져오기 (CUDA_VISIBLE_DEVICES 설정 전에)
                gpu_name = torch.cuda.get_device_name(gpu_id)
                gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                
                # 특정 GPU 설정
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                print(f"✅ GPU {gpu_id} 사용 설정 완료")
                print(f"   GPU 이름: {gpu_name}")
                print(f"   GPU 메모리: {gpu_memory:.1f}GB")
                
                # CUDA_VISIBLE_DEVICES 설정 후에는 device를 "0"으로 변경
                args.device = "0"
                
            except ValueError:
                print(f"❌ 올바르지 않은 device 값: {args.device}. '0', '1', '2', '3' 또는 'cpu'를 사용하세요.")
                args.device = "0"
            except Exception as e:
                print(f"⚠️  GPU 설정 중 오류: {e}")
                print("GPU 0으로 기본 설정합니다.")
                args.device = "0"
    
    print(f"🚀 사용할 디바이스: {args.device}")
    
    # W&B 초기화 (train_simple.py 스타일)
    wandb_run = None
    if not args.no_wandb:
        try:
            # W&B run name 설정
            if args.wandb_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                config_suffix = "_custom_cfg" if args.cfg else ""
                args.wandb_name = f"{args.name}{config_suffix}_{timestamp}"
            
            # W&B config 준비
            wandb_config = {
                "model": args.model,
                "config_file": args.cfg if args.cfg else "default",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "image_size": args.imgsz,
                "device": args.device,
                "workers": args.workers,
                "patience": args.patience,
                "optimizer": "AdamW",
                "dataset": "KAIST RGB-T",
                "architecture": "YOLO11n",
                "input_channels": 6,  # RGB-T
                "gpu_name": gpu_name,
                "gpu_memory_gb": gpu_memory,
            }
            
            # Config 파일이 있으면 해당 정보도 추가
            if args.cfg:
                wandb_config["custom_config"] = True
                wandb_config["config_path"] = args.cfg
            
            # W&B 초기화
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.entity,
                name=args.wandb_name,
                config=wandb_config,
                tags=["YOLO11n", "RGB-T", "KAIST", "object-detection"] + (["custom-config"] if args.cfg else []),
                notes=f"YOLO11n training on KAIST RGB-T dataset with 6-channel input{' using custom config' if args.cfg else ''}",
                resume="allow"
            )
            print(f"✅ W&B 연결 완료: {wandb_run.url}")
            
        except Exception as e:
            print(f"⚠️  W&B 초기화 실패: {e}")
            print("W&B 없이 훈련을 계속합니다.")
            wandb_run = None
    else:
        print("ℹ️  W&B 로깅이 비활성화되었습니다.")
    
    # 모델 로드 (Config 파일 지원)
    if args.cfg:
        print(f"🔧 사용자 정의 Config로 모델 초기화: {args.cfg}")
        print(f"📁 사전 훈련된 가중치 로딩: {args.model}")
        
        # Config 파일로 모델 구조 초기화
        model = YOLO(args.cfg, task='detect')
        
        # 사전 훈련된 가중치가 있다면 로드 (선택적)
        if args.model and args.model != "yolo11n.pt" and os.path.exists(args.model):
            print(f"✅ 사전 훈련된 가중치 적용: {args.model}")
            try:
                # 가중치만 로드 (구조는 config에서 가져온 것을 유지)
                checkpoint = torch.load(args.model, map_location='cpu')
                if 'model' in checkpoint:
                    weights = checkpoint['model']
                else:
                    weights = checkpoint
                    
                # 호환 가능한 레이어만 로드
                model.model.load_state_dict(weights, strict=False)
                print("✅ 사전 훈련된 가중치 일부 적용 완료")
            except Exception as e:
                print(f"⚠️  가중치 로드 실패 (config 모델로 진행): {e}")
        else:
            print("ℹ️  Config 파일로만 모델을 초기화합니다 (사전 훈련된 가중치 없음)")
    else:
        print(f"📁 사전 훈련된 모델 로딩: {args.model}")
        model = YOLO(args.model)
    
    # RGB-T 입력을 위한 모델 수정 (6채널)
    print("🔧 RGB-T 6채널 입력을 위한 모델 확인 및 수정 중...")
    
    modified_channels = False
    try:
        if hasattr(model.model, 'model') and len(model.model.model) > 0:
            first_layer = model.model.model[0]
            if hasattr(first_layer, 'conv'):
                first_conv = first_layer.conv
                current_channels = first_conv.in_channels
                
                print(f"ℹ️  현재 입력 채널 수: {current_channels}")
                
                if current_channels == 3:
                    print("🔄 3채널 → 6채널 변환 중...")
                    new_conv = torch.nn.Conv2d(
                        in_channels=6,
                        out_channels=first_conv.out_channels,
                        kernel_size=first_conv.kernel_size,
                        stride=first_conv.stride,
                        padding=first_conv.padding,
                        bias=first_conv.bias is not None
                    )
                    
                    # 기존 RGB 가중치 복사 및 확장
                    with torch.no_grad():
                        new_conv.weight[:, :3] = first_conv.weight
                        new_conv.weight[:, 3:] = first_conv.weight  # Thermal 채널에 RGB 가중치 복사
                        if first_conv.bias is not None:
                            new_conv.bias = first_conv.bias
                    
                    model.model.model[0].conv = new_conv
                    modified_channels = True
                    print("✅ 모델이 RGB-T 6채널 입력으로 수정되었습니다.")
                    
                elif current_channels == 6:
                    print("✅ 모델이 이미 6채널 입력으로 설정되어 있습니다.")
                else:
                    print(f"⚠️  예상치 못한 입력 채널 수: {current_channels}")
                
                # W&B에 모델 아키텍처 로깅
                if wandb_run:
                    wandb.log({
                        "model_input_channels": 6 if modified_channels or current_channels == 6 else current_channels,
                        "model_modification": "RGB-T_6_channel" if modified_channels else "no_modification",
                        "original_channels": current_channels,
                        "final_channels": 6 if modified_channels or current_channels == 6 else current_channels,
                        "channels_modified": modified_channels
                    })
            else:
                print("⚠️  첫 번째 레이어에서 conv 속성을 찾을 수 없습니다.")
        else:
            print("⚠️  모델 구조에 접근할 수 없습니다.")
            
    except Exception as e:
        print(f"⚠️  모델 수정 중 오류 발생: {e}")
        print("기본 모델 구조를 유지합니다.")
    
    # 훈련 설정 출력
    print(f"""
🔥 훈련 설정:
   📊 데이터셋: {args.data}
   🏗️  모델 Config: {args.cfg if args.cfg else '기본 설정'}
   🏃 에포크: {args.epochs}
   📦 배치 크기: {args.batch_size}
   🖼️  이미지 크기: {args.imgsz}x{args.imgsz}
   💻 디바이스: {args.device} ({gpu_name})
   👷 워커 수: {args.workers}
   ⏰ 조기 종료 patience: {args.patience}
   💾 저장 주기: {args.save_period} 에포크마다
   📈 W&B 프로젝트: {args.wandb_project if not args.no_wandb else 'Disabled'}
    """)
    
    # 훈련 시작
    print("🚀 훈련을 시작합니다...")
    
    try:
        training_args = {
            'data': args.data,
            'epochs': args.epochs,
            'batch': args.batch_size,
            'imgsz': args.imgsz,
            'device': args.device,
            'project': args.project,
            'name': args.name,
            'workers': args.workers,
            'patience': args.patience,
            'save': True,
            'plots': True,
            'val': True,
            'verbose': True,
            'seed': 0,
            'deterministic': True,
            'resume': args.resume if args.resume else False,
            'save_period': args.save_period,
            
            # Optimizer settings
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Loss settings
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            
            # Augmentation settings
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
        }
        
        # Config 파일이 있는 경우 cfg 인자 추가
        if args.cfg:
            training_args['cfg'] = args.cfg
        
        results = model.train(**training_args)
        
        # 훈련 완료 후 W&B에 최종 결과 로깅
        if wandb_run and results:
            final_metrics = {
                "final/best_mAP50": getattr(results, 'results_dict', {}).get('metrics/mAP50(B)', 0),
                "final/best_mAP50-95": getattr(results, 'results_dict', {}).get('metrics/mAP50-95(B)', 0),
                "final/best_precision": getattr(results, 'results_dict', {}).get('metrics/precision(B)', 0),
                "final/best_recall": getattr(results, 'results_dict', {}).get('metrics/recall(B)', 0),
                "training_completed": True,
                "total_epochs": args.epochs,
                "used_custom_config": bool(args.cfg)
            }
            wandb.log(final_metrics)
            
            # 모델 아티팩트 업로드 (선택사항)
            try:
                best_model_path = f"{args.project}/{args.name}/weights/best.pt"
                if os.path.exists(best_model_path):
                    wandb.save(best_model_path, base_path=args.project)
                    print("✅ 최고 성능 모델을 W&B에 업로드했습니다.")
                    
                # Config 파일도 함께 업로드
                if args.cfg and os.path.exists(args.cfg):
                    wandb.save(args.cfg)
                    print("✅ 사용된 Config 파일을 W&B에 업로드했습니다.")
                    
            except Exception as e:
                print(f"⚠️  파일 업로드 실패: {e}")
        
        print(f"✅ 훈련 완료! 최고 결과: {results}")
        
    except Exception as e:
        print(f"❌ 훈련 중 오류 발생: {e}")
        if wandb_run:
            wandb.log({"training_error": str(e), "training_completed": False})
        raise
    
    finally:
        # W&B 세션 종료
        if wandb_run:
            wandb.finish()
            print("📊 W&B 세션이 종료되었습니다.")
    
    return results

if __name__ == "__main__":
    main()