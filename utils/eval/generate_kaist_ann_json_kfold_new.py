import os
import json
import xml.etree.ElementTree as ET


def convert_val_fold_to_json(
    val_txt_path='data/folds/fold1/val.txt',
    xml_ann_dir='datasets/kaist-rgbt/train/labels-xml',
    output_json_path='utils/eval/kaist_val_fold1.json'
):
    categories = [
        {"id": 0, "name": "person"},
        {"id": 1, "name": "cyclist"},
        {"id": 2, "name": "people"},
        {"id": 3, "name": "person?"}
    ]

    with open(val_txt_path, 'r') as f:
        image_list = [line.strip() for line in f.readlines()]

    images = []
    annotations = []
    image_name_to_id = {}
    image_id = 0
    ann_id = 0

    for image_path in image_list:
        image_name = os.path.basename(image_path).replace('.jpg', '')
        xml_file = os.path.join(xml_ann_dir, image_name + '.xml')

        images.append({
            "id": image_id,
            "im_name": image_name + ".jpg",
            "height": 512,
            "width": 640
        })
        image_name_to_id[image_name] = image_id

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

                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "height": h,
                    "occlusion": occlusion,
                    "ignore": difficult
                })
                ann_id += 1

        image_id += 1

    coco_dict = {
        "info": {
            "dataset": "KAIST Multispectral Pedestrian Benchmark",
            "url": "https://soonminhwang.github.io/rgbt-ped-detection/",
            "publish": "CVPR 2015"
        },
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(coco_dict, f, indent=4)

    print(f"[✅] 저장 완료: {output_json_path}")
    print(f"  - 이미지 수: {len(images)}")
    print(f"  - 어노테이션 수: {len(annotations)}")




# 예시 호출
if __name__ == "__main__":
    val_txt = "/home/lbw/workspace/AUE8088/data/folds/fold1/val.txt"
    xml_dir = "datasets/kaist-rgbt/train/labels-xml"
    output_json = "utils/eval/kaist_val_fold1.json"
    convert_val_fold_to_json(val_txt, xml_dir, output_json)
