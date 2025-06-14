import os
import json
import yaml
import argparse
import xml.etree.ElementTree as ET


def convert_from_data_yaml(yaml_path, xml_ann_dir, output_json_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    val_txt_path = data['val']
    if isinstance(val_txt_path, list):
        val_txt_path = val_txt_path[0]

    print(f"[INFO] val.txt: {val_txt_path}")
    print(f"[INFO] xmlAnnDir: {xml_ann_dir}")
    print(f"[INFO] output_json: {output_json_path}")

    with open(val_txt_path, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]

    categories = [
        {"id": 0, "name": "person"},
        {"id": 1, "name": "cyclist"},
        {"id": 2, "name": "people"},
        {"id": 3, "name": "person?"}
    ]

    kaist_annotation = {
        "info": {
            "dataset": "KAIST Multispectral Pedestrian Benchmark",
            "url": "https://soonminhwang.github.io/rgbt-ped-detection/",
            "related_project_url": "http://multispectral.kaist.ac.kr",
            "publish": "CVPR 2015"
        },
        "images": [],
        "annotations": [],
        "categories": categories
    }

    image_id = 0
    annotation_id = 0

    for path in image_paths:
        image_name = os.path.basename(path).replace('.jpg', '')
        xml_file = os.path.join(xml_ann_dir, image_name + '.xml')

        kaist_annotation['images'].append({
            "id": image_id,
            "im_name": image_name + '.jpg',
            "height": 512,
            "width": 640
        })

        if os.path.exists(xml_file):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for obj in root.findall('object'):
                name = obj.find('name').text
                category_id = next((c['id'] for c in categories if c['name'] == name), None)
                if category_id is None:
                    continue

                bbox = obj.find('bndbox')
                x = int(bbox.find('x').text)
                y = int(bbox.find('y').text)
                w = int(bbox.find('w').text)
                h = int(bbox.find('h').text)
                difficult = int(obj.find('difficult').text)
                occlusion = int(obj.find('occlusion').text)

                kaist_annotation['annotations'].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "height": h,
                    "occlusion": occlusion,
                    "ignore": difficult
                })
                annotation_id += 1

        image_id += 1

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(kaist_annotation, f, indent=4)

    print(f"[✅] 저장 완료: {output_json_path}")
    print(f"  - 이미지 수: {len(kaist_annotation['images'])}")
    print(f"  - 어노테이션 수: {len(kaist_annotation['annotations'])}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, required=True, help="학습용 data.yaml 파일 경로")
    parser.add_argument("--xmlAnnDir", type=str, required=True, help="XML 어노테이션 디렉토리")
    parser.add_argument("--jsonOut", type=str, required=True, help="출력 JSON 파일 경로")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_opt()
    convert_from_data_yaml(args.yaml, args.xmlAnnDir, args.jsonOut)
