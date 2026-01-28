# ============================================================
# settings.py
#
# 역할:
# - 프로젝트 전반에서 사용하는 "경로"만 중앙 관리
# - 알고리즘 / 좌표계 / 파라미터 로직과 완전 분리
#
# 원칙:
# - 계산 결과에 영향 없음
# - detector / pipeline / coordinate는 참조만 한다
# - 경로 변경은 이 파일에서만 수행
# ============================================================


# =========================
# 📂 프로젝트 구성 요약
# =========================
#
# main.py
#   - 프로그램 실행 엔트리
#   - realsense_loop.run() 호출
#   - 각 모듈 콜백 연결(배선 역할)
#
# realsense_loop.py
#   - RealSense 카메라 루프
#   - color / depth 프레임 수신
#   - 키 입력(space / r / esc) 처리
#   - 화면 표시 + 마우스 콜백 연결
#
# click_points.py
#   - 마우스 클릭 좌표 관리
#   - 스냅샷 시점 클릭 포인트를 world 좌표로 변환
#   - Save_Cam() 제공
#
# detector.py
#   - 물체 자동 검출 로직
#   - green : depth + DBSCAN + 회전 박스
#   - blue  : RGB 윤곽 + depth hole
#   - run() -> (vis, items)
#
# depth_utils.py
#   - depth 관련 유틸
#   - FakeDepthFrameFromNpy
#   - 박스 중심 계산
#   - 파랑 객체용 안전 depth 탐색
#
# coordinate.py
#   - pixel + depth -> world 좌표 변환
#   - camcalib.npz 로드
#   - 5x5 depth median + 보정 적용
#
# pipeline.py
#   - 스페이스바 시 실행되는 핵심 파이프라인
#   - 저장 → 검출 → world 계산 → flat list 생성
#
# settings.py
#   - 경로 정의 전용 (이 파일)
#
# outputs/
#   - 실행 결과 저장 폴더
#   - color.jpg / depth.npy
# =========================


# =========================
# 📁 경로 설정
# =========================
import os

# 프로젝트 루트 기준 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 출력 디렉토리
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# 결과 파일
COLOR_PATH = os.path.join(OUTPUT_DIR, "color.jpg")   # 시각화 결과
DEPTH_PATH = os.path.join(OUTPUT_DIR, "depth.npy")   # depth snapshot (z16, mm)

# 캘리브레이션 파일
CALIB_PATH = os.path.join(BASE_DIR, "camcalib.npz")
