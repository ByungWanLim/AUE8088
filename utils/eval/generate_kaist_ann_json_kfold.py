import argparse
import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def convert_ann_xml2json(textListFile, xmlAnnDir, jsonAnnFile):
    """단일 폴드의 어노테이션 XML을 JSON으로 변환"""
    print(f"처리 중: {textListFile} -> {jsonAnnFile}")
    
    with open(textListFile, 'r') as f:
        lines = f.readlines()

    kaist_annotation = {
        "info": {
            "dataset": "KAIST Multispectral Pedestrian Benchmark",
            "url": "https://soonminhwang.github.io/rgbt-ped-detection/",
            "related_project_url": "http://multispectral.kaist.ac.kr",
            "publish": "CVPR 2015"
        },
        "info_improved": {
            "sanitized_annotation": {
                "publish": "BMVC 2018",
                "url": "https://li-chengyang.github.io/home/MSDS-RCNN/",
                "target": "files in train-all-02.txt (set00-set05)"
            },
            "improved_annotation": {
                "url": "https://github.com/denny1108/multispectral-pedestrian-py-faster-rcnn",
                "publish": "BMVC 2016",
                "target": "files in test-all-20.txt (set06-set11)"
            }
        },
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "person"},
            {"id": 1, "name": "cyclist"},
            {"id": 2, "name": "people"},
            {"id": 3, "name": "person?"}
        ]
    }

    image_id = 0
    annotation_id = 0

    for line in lines:
        image_path = line.strip()
        image_name = image_path.split('/')[-1].replace('.jpg', '')
        annotation_file = os.path.join(xmlAnnDir, image_name + '.xml')

        kaist_annotation['images'].append({
            "id": image_id,
            "im_name": image_name + '.jpg',
            "height": 512,
            "width": 640
        })

        if os.path.exists(annotation_file):
            tree = ET.parse(annotation_file)
            root = tree.getroot()

            for obj in root.findall('object'):
                category_name = obj.find('name').text
                category_id = next((item['id'] for item in \
                                    kaist_annotation['categories'] if item["name"] == category_name), None)
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('x').text)
                ymin = int(bbox.find('y').text)
                width = int(bbox.find('w').text)
                height = int(bbox.find('h').text)
                occlusion = int(obj.find('occlusion').text)
                ignore = int(obj.find('difficult').text)

                kaist_annotation['annotations'].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [xmin, ymin, width, height],
                    "height": height,
                    "occlusion": occlusion,
                    "ignore": ignore
                })

                annotation_id += 1

        image_id += 1

    # 디렉토리가 없다면 생성
    os.makedirs(os.path.dirname(jsonAnnFile), exist_ok=True)
    
    with open(jsonAnnFile, 'w') as f:
        json.dump(kaist_annotation, f, indent=4)
    
    print(f"생성 완료: {jsonAnnFile} (이미지 {image_id}개, 어노테이션 {annotation_id}개)")


def create_kfold_annotations(fold_dir, n_folds, xmlAnnDir, output_dir):
    """k-fold 교차 검증을 위한 어노테이션 JSON 파일 생성"""
    fold_dir = Path(fold_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for fold in range(1, n_folds + 1):
        val_file = fold_dir / f'fold{fold}' / 'val.txt'
        if not val_file.exists():
            print(f"경고: {val_file} 파일이 존재하지 않습니다.")
            continue
        
        json_file = output_dir / f'KAIST_val-A_annotation_fold{fold}.json'
        convert_ann_xml2json(str(val_file), xmlAnnDir, str(json_file))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_dir", type=str, default='data/folds', 
                        help="폴드 데이터가 저장된 디렉토리 경로")
    parser.add_argument("--n_folds", type=int, default=5, 
                        help="폴드 수 (기본값: 5)")
    parser.add_argument("--xmlAnnDir", type=str, default='datasets/kaist-rgbt/train/labels-xml', 
                        help="XML 어노테이션 디렉토리")
    parser.add_argument("--output_dir", type=str, default='utils/eval/folds', 
                        help="출력 JSON 어노테이션 디렉토리")
    parser.add_argument("--single_fold", type=int, default=0, 
                        help="단일 폴드만 처리 (0: 모든 폴드, 1-5: 특정 폴드)")
    
    # 단일 파일 변환을 위한 기존 옵션 유지
    parser.add_argument("--textListFile", type=str, default='', 
                        help="이미지 파일명이 포함된 텍스트 파일 (예: val-D.txt)")
    parser.add_argument("--jsonAnnFile", type=str, default='', 
                        help="출력 JSON 파일명")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_opt()
    
    # 단일 파일 변환 모드
    if args.textListFile and args.jsonAnnFile:
        convert_ann_xml2json(args.textListFile, args.xmlAnnDir, args.jsonAnnFile)
    # k-fold 모드
    else:
        if args.single_fold > 0:
            # 단일 폴드만 처리
            val_file = os.path.join(args.fold_dir, f'fold{args.single_fold}', 'val.txt')
            json_file = os.path.join(args.output_dir, f'KAIST_val-A_annotation_fold{args.single_fold}.json')
            convert_ann_xml2json(val_file, args.xmlAnnDir, json_file)
        else:
            # 모든 폴드 처리
            create_kfold_annotations(args.fold_dir, args.n_folds, args.xmlAnnDir, args.output_dir)
    
    print("어노테이션 생성 완료!")