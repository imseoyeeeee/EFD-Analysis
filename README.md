## 실행
python -m EFD-Analysis.main \
  --cfgs "/home/user/GaitRecognition/EFD-Analysis/configs/fineefd.yaml" \
  --vis no

## 설정
--vis: no로 작성하면 실행되지 않습니다
denoising: config의 denoise_config.enable (기본 false)로 선택 (프레임 수 제각각)

efd_config
- normalize: false (기본) - 회전(팔다리 흔들림 각도)과 part 간 상대 크기 유지, 출력 폴더 EFD{n}-unnorm
  - align_start_point: true - 윤곽 시작점에 따른 계수 차이만 제거
  - scale_to_height: true - 프레임별 몸 높이로 나눔 (카메라 거리 보정)
- normalize: true - pyefd 정규화 (회전/크기/시작점 불변), 출력 폴더 EFD{n}

- skip_existing: true - 이미 저장된 pkl은 건너뛰고 이어서 추출 (중단된 작업 재개)

## 추출 결과 검사 / 정리
python EFD-Analysis/data_check.py --cfgs EFD-Analysis/configs/fineefd.yaml --num 20
- 무작위 pkl 몇 개의 part별 존재 비율을 출력하고, 이진 실루엣(0/255, 0/1)이나 컬러 이미지를 넣은 경우, 읽을 수 없는 파일, harmonic 수가 섞인 경우를 알려 줍니다
- --cfgs 대신 --root <EFD 폴더>로 폴더를 직접 지정할 수도 있습니다

python EFD-Analysis/trim_harmonics.py --root <EFD 폴더> --harmonics 32
- 이미 추출한 pkl의 harmonic을 앞 N개만 남기고, 읽을 수 없는 pkl은 삭제합니다

## configs
기존 8분할 - fineefd.yaml
상지/하지 - coarseefd.yaml
config내에서 데이터 경로 및 디노이징 설정 수정 가능합니다

## codes
EFD.py - img > efd 변환함수
KillNoise.py - 노이즈 제거함수
