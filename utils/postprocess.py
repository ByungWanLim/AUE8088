import torch
import numpy as np
from utils.general import non_max_suppression, box_iou

def handle_people_class(
    pred, 
    conf_thres=0.25, 
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300
):
    """
    "people"(class_id=2)과 "person"(class_id=0) 간의 관계를 처리하는 후처리 함수
    
    Args:
        pred (list): 모델 예측 결과 [nx6 텐서 리스트(x1, y1, x2, y2, conf, cls)]
        conf_thres (float): 신뢰도 임계값
        iou_thres (float): NMS IoU 임계값
        classes (list): 필터링할 클래스 ID 목록 (None=모든 클래스)
        agnostic (bool): 클래스 비의존적 NMS 사용 여부
        multi_label (bool): 다중 레이블 할당 여부
        labels (tuple): DB 레이블 (unused)
        max_det (int): 이미지당 최대 검출 수
        
    Returns:
        list: 처리된 검출 결과 리스트
    """
    # 1. 기본 NMS 적용
    output = non_max_suppression(
        pred, conf_thres, iou_thres, classes, agnostic, multi_label, labels, max_det
    )
    
    # 후처리된 결과 저장
    processed_output = []
    
    # 2. 각 이미지별 결과 처리
    for i, det in enumerate(output):
        if len(det) == 0:
            processed_output.append(det)
            continue
            
        # 검출 결과 복사
        processed_det = det.clone()
        
        # 3. "people" 클래스(ID: 2)와 "person" 클래스(ID: 0) 간의 관계 처리
        # 클래스별로 박스 분리
        people_mask = processed_det[:, 5] == 2
        person_mask = processed_det[:, 5] == 0
        
        if people_mask.any() and person_mask.any():
            people_boxes = processed_det[people_mask]
            person_boxes = processed_det[person_mask]
            
            # "people" 박스와 "person" 박스 간의 중첩 처리
            # 각 "people" 박스에 대해
            for p_idx in range(people_boxes.shape[0]):
                p_box = people_boxes[p_idx, :4]  # 좌표만 추출
                p_conf = people_boxes[p_idx, 4]  # 신뢰도
                
                # 각 "person" 박스와의 IoU 계산
                ious = box_iou(p_box.unsqueeze(0), person_boxes[:, :4])
                
                # 중첩된 "person" 박스들 (IoU > threshold)
                overlap_mask = ious[0] > 0.7
                
                if overlap_mask.any():
                    overlapped_persons = person_boxes[overlap_mask]
                    
                    # 대응 방법 1: "people" 박스가 높은 신뢰도를 가진 경우, 중첩된 "person" 박스들의 신뢰도 조정
                    if p_conf > 0.6:
                        # "person" 박스들의 신뢰도 가중치 계산 (IoU 기반)
                        weight = torch.clamp(1.0 - (ious[0][overlap_mask] - 0.7) / 0.3, 0.5, 1.0)
                        
                        # "person" 박스들의 신뢰도 조정
                        person_boxes[overlap_mask, 4] *= weight
                    
                    # 대응 방법 2: "people" 박스가 낮은 신뢰도를 가진 경우, "people" 박스 신뢰도 조정
                    else:
                        # 중첩된 "person" 박스들의 최대 신뢰도 확인
                        max_person_conf = overlapped_persons[:, 4].max()
                        
                        # "person" 박스들이 더 확실한 경우, "people" 박스 신뢰도 조정
                        if max_person_conf > p_conf:
                            people_boxes[p_idx, 4] *= 0.8  # "people" 박스 신뢰도 감소
            
            # 조정된 신뢰도 적용
            processed_det[people_mask] = people_boxes
            processed_det[person_mask] = person_boxes
            
            # 신뢰도 임계값 재적용
            conf_mask = processed_det[:, 4] >= conf_thres
            processed_det = processed_det[conf_mask]
        
        processed_output.append(processed_det)
    
    return processed_output