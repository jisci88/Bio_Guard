# Raspberry Pi 5 + Hailo-8 얼굴 추적 설치·검증 가이드

이 문서는 **Hailo-8(26 TOPS, `HAILO8`)** 전용이다. Hailo-8L(`HAILO8L`)은 대상이 아니다. 현재 장비가 보고한 기준은 펌웨어/HailoRT 4.23 계열(펌웨어 4.23.0)이다. 모터를 연결하기 전에 반드시 이 문서의 읽기 전용 진단, 모델 확인, probe, 하드웨어 unittest 순서로 검증한다.

## 1. 구조와 안전 경계

`HailoHybridFaceDetector`는 두 모델을 비동기로 함께 실행한다.

- SCRFD는 실제 얼굴의 정확한 박스를 제공한다. 화면 안쪽의 신선한 `SCRFD`만 생체 신호 센서 ROI(`sensor_box`)에 쓴다.
- 화면 가장자리의 `SCRFD_EDGE`, optical-flow `FLOW`, YOLO person/head의 `PERSON_HEAD`는 팬/틸트가 사용자를 다시 중앙에 두기 위한 모터 박스 전용이다. 이 폴백들은 센서 ROI가 아니다.
- YOLO person/head는 얼굴이 빠르게 움직이거나 프레임 밖으로 나간 동안 복구 목표를 유지한다. 불확실하거나 빈 화면이면 `NONE`이며 새 모터 목표를 만들면 안 된다.

`tools/hailo_face_probe.py`는 카메라/이미지와 detector만 소유한다. Dynamixel, GPIO, 부저, 열화상, anomaly, 측정 pipeline을 import하거나 초기화하지 않는다.

## 2. 요구 사항과 호환성

| 항목 | 대회 기준 | 확인/주의 |
|---|---|---|
| 호스트 | Raspberry Pi 5, 64-bit Raspberry Pi OS | `uname -m`이 `aarch64`인지 확인 |
| 가속기 | Hailo-8 / `HAILO8` | Hailo-8L HEF와 장치를 사용하지 않음 |
| 펌웨어·HailoRT | 장비 보고값 4.23 계열, 펌웨어 4.23.0 허용 | patch 버전을 probe가 보고하지만 `==4.23.0`으로 차단하지 않음 |
| 드라이버 | 설치된 HailoRT 4.23과 호환되는 PCIe driver | driver/runtime/firmware 버전은 상호 호환되어야 함 |
| Apps/TAPPAS | 설치된 4.23 환경에 맞는 Hailo Apps/TAPPAS | 다른 세대의 최신판을 무조건 덮어쓰지 않음 |
| Python | HailoRT binding과 Hailo Apps helper가 같은 `python3`에서 import 가능 | 보통 apt binding은 `/usr/lib/python3/dist-packages`, Apps venv는 `setup_env.sh`로 활성화 |
| 모델 | `scrfd_10g` + 공식 COCO `yolov8m`, 모두 HAILO8용 | 파일명만 바꿔 architecture를 바꿀 수 없음 |

Hailo의 현재 설치 문서는 Hailo-8/8L에 HailoRT 4.23과 TAPPAS Core 5.1.0 조합을 명시하고, runtime package가 Apps보다 먼저 설치되어야 한다고 설명한다. [Hailo Apps 공식 설치 문서](https://github.com/hailo-ai/hailo-apps/blob/main/doc/user_guide/installation.md)와 [Hailo Apps Core 호환성 표](https://github.com/hailo-ai/hailo-apps-core)를 기준으로 한다. Raspberry Pi의 공식 절차는 AI Kit/AI HAT+에 `dkms`, `hailo-all`, reboot, `hailortcli fw-control identify` 순서를 제시한다. [Raspberry Pi AI software 공식 문서](https://www.raspberrypi.com/documentation/computers/ai.html)

이미 4.23 장비가 동작한다면 먼저 아래 진단 결과를 저장하고, 단순히 “최신”으로 올리기 위해 `apt full-upgrade`나 Apps `main` 설치를 실행하지 않는다. 새 설치라면 공식 Raspberry Pi 절차는 다음과 같지만, 4.23을 고정해야 할 때는 `apt-cache policy`에 실제 제공되는 정확한 package revision을 확인한 뒤 서로 맞는 세트를 설치한다.

```bash
sudo apt update
apt-cache policy hailo-all hailort hailo-dkms python3-hailort hailo-tappas-core
sudo apt install dkms
sudo apt install hailo-all
sudo reboot
```

## 3. 변경 없는 사전 진단

프로젝트 루트에서 결과를 그대로 보관한다.

```bash
uname -a
uname -m
hailortcli fw-control identify
hailortcli scan
hailortcli --version
dpkg-query -W 'hailo*' 2>&1
```

`identify`에서 `Device Architecture: HAILO8`(일부 도구는 `HAILO8_B0`)과 firmware를 확인한다. `HAILO8L`이면 중단한다. Hailo가 제시한 identify 출력 예시는 Board Name과 Device Architecture를 함께 보여 준다. [HailoRT 공식 검증 문서](https://github.com/hailo-ai/hailo-apps-core/blob/master/docs/installation/verify_hailoRT.rst)

Apps 환경을 활성화한 터미널에서 Python 경로와 버전을 확인한다.

```bash
command -v python3
python3 -c "import sys; print(sys.executable); print(*sys.path, sep='\n')"
python3 -c "import importlib.metadata as m; print('hailort', m.version('hailort')); print('hailo-apps', m.version('hailo-apps'))"
python3 -c "import hailo_platform; print('hailo_platform', hailo_platform.__file__)"
python3 -c "from hailo_apps.python.core.common.hailo_inference import HailoInfer; print('preferred HailoInfer OK')"
```

preferred import가 실패하면 해당 Apps checkout에서 먼저 `source setup_env.sh`를 실행하고 다시 확인한다. 구 Hailo Apps release는 아래 fallback 경로 중 하나에 `HailoInfer`를 제공할 수 있다. 프로젝트 detector도 preferred 경로 다음에 이 두 경로를 순서대로 탐색한다.

```bash
python3 -c "from hailo_apps_infra.hailo_inference import HailoInfer; print('legacy hailo_inference HailoInfer OK')"
python3 -c "from hailo_apps_infra.hailo_rpi_common import HailoInfer; print('legacy hailo_rpi_common HailoInfer OK')"
```

HEF와 label sidecar를 읽기 전용으로 찾는다.

```bash
find /usr/local/hailo/resources /usr/share/hailo-apps /opt/hailo "$HOME" \
  -type f \( -iname '*.hef' -o -iname '*.json' -o -iname '*.txt' -o -iname '*.yaml' -o -iname '*.yml' \) \
  2>/dev/null | sort
find /usr/local/hailo/resources/models/hailo8 -type f 2>/dev/null | sort
```

## 4. 공식 resource 확인과 모델 배치

현재 Hailo Apps는 architecture별 모델을 `/usr/local/hailo/resources/models/hailo8/` 아래에 놓고 `hailo-download-resources`로 목록/preview/download를 제공한다. [공식 resource 명령과 옵션](https://github.com/hailo-ai/hailo-apps/blob/main/doc/user_guide/installation.md#download-resources), [공식 resource directory 설정](https://github.com/hailo-ai/hailo-apps/blob/main/hailo_apps/config/config.yaml)

설치된 release가 이 CLI를 제공할 때만 다음 순서로 실행한다. 먼저 목록과 dry-run으로 실제 이름/architecture를 확인한다.

```bash
source /path/to/hailo-apps/setup_env.sh
hailo-download-resources --arch hailo8 --list-models
hailo-download-resources --arch hailo8 --group face_recognition --dry-run
hailo-download-resources --arch hailo8 --group face_recognition
find /usr/local/hailo/resources/models/hailo8 -type f | sort
```

`scrfd_10g`와 `yolov8m`이 `--list-models`에 실제로 표시될 때 다음처럼 각각 받는다. 설치된 CLI는 개별 다운로드에 `--resource-name`과 소속 `--group`을 함께 요구한다.

```bash
sudo mkdir -p /usr/local/hailo/resources/models/hailo8
sudo chown -R "$(id -un)":"$(id -gn)" /usr/local/hailo
hailo-download-resources --arch hailo8 --group face_recognition --resource-name scrfd_10g --resource-type model --dry-run
hailo-download-resources --arch hailo8 --group face_recognition --resource-name scrfd_10g --resource-type model
hailo-download-resources --arch hailo8 --group detection --resource-name yolov8m --resource-type model --dry-run
hailo-download-resources --arch hailo8 --group detection --resource-name yolov8m --resource-type model
```

목록에 없다면 URL을 추측하지 않는다. 설치된 release의 resource 결과 또는 Hailo Developer Zone/Model Explorer에서 **HAILO8용** artifact를 확보해 수동 배치한다. Hailo Model Zoo는 모델 변환/컴파일 도구이며, target architecture가 다른 HEF는 호환되지 않는다. [Hailo Model Zoo 공식 저장소](https://github.com/hailo-ai/hailo_model_zoo)

프로젝트의 최종 계약은 다음과 같다.

```text
models/hailo8/
├── scrfd_10g.hef
└── yolov8m.hef
```

공식 `yolov8m`은 COCO class 0=`person` 계약으로 전신/머리 복구만 담당하고, 얼굴 ROI는 SCRFD가 담당한다. detector는 정확한 `yolov8m` stem에만 이 내장 계약을 허용하며 나머지 79개 COCO class는 무시한다. 선택적으로 custom `yolov8n_personface.hef`를 사용하려면 동일 stem의 sidecar에 class 0=`person`, class 1=`face`가 있어야 한다. HAILO8L binary를 이름만 바꿔 사용하지 않는다.

수동 복사 예시(확인한 source path만 넣는다):

```bash
mkdir -p models/hailo8
cp /verified/hailo8/path/scrfd_10g.hef models/hailo8/scrfd_10g.hef
cp /verified/hailo8/path/yolov8m.hef models/hailo8/yolov8m.hef
```

## 5. 모터 연결 전 probe와 unittest

고정 이미지가 가장 재현성이 높다. probe는 이미지를 RGB uint8로 반복하고, empty scene의 `NONE`은 정상으로 허용한다.

```bash
python3 tools/hailo_face_probe.py \
  --image /absolute/path/visible-face.jpg \
  --scrfd-hef "$PWD/models/hailo8/scrfd_10g.hef" \
  --personface-hef "$PWD/models/hailo8/yolov8m.hef" \
  --frames 100 --warmup 20 --require-face
```

카메라는 Picamera2/libcamera를 우선하고, import/open이 실패하면 OpenCV camera 0으로 폴백한다.

```bash
python3 tools/hailo_face_probe.py --camera 0 \
  --scrfd-hef "$PWD/models/hailo8/scrfd_10g.hef" \
  --personface-hef "$PWD/models/hailo8/yolov8m.hef" \
  --frames 300 --warmup 30 --require-face
```

출력에서 절대 HEF 경로, HAILO8 identity, input shape, 노출된 output vstream name/shape/type, label path, priorities, 두 모델 completion, callback/decode error 0, shutdown diagnostics를 확인한다. 빈 장면 시험에서는 `--require-face`를 빼도 된다.

그 다음 opt-in 하드웨어 unittest를 실행한다. black frame은 `NONE`이어도 성공이며 두 모델 completion과 callback 정상 여부를 검증한다.

```bash
BIO_HAILO_HARDWARE=1 \
BIO_HAILO_SCRFD_HEF="$PWD/models/hailo8/scrfd_10g.hef" \
BIO_HAILO_PERSONFACE_HEF="$PWD/models/hailo8/yolov8m.hef" \
python3 -m unittest tests.test_hailo_hardware -v
```

충분히 크게 보이는 얼굴 fixture까지 검증:

```bash
BIO_HAILO_HARDWARE=1 \
BIO_HAILO_SCRFD_HEF="$PWD/models/hailo8/scrfd_10g.hef" \
BIO_HAILO_PERSONFACE_HEF="$PWD/models/hailo8/yolov8m.hef" \
BIO_HAILO_FACE_IMAGE=/absolute/path/visible-face.jpg \
python3 -m unittest tests.test_hailo_hardware -v
```

Windows local unittest는 import, mock, cleanup, 요약 로직만 검증한다. Pi의 Hailo device, camera, 실제 HEF output/decode, throughput 검증을 대체하지 않는다.

## 6. production 실행

기존 camera/Dynamixel 방향 인자는 그대로이며 detector 기본값은 `hailo-hybrid`이다. probe와 hardware unittest가 모두 통과한 뒤 실행한다.

```bash
python3 vital_monitor.py --camera 0 \
  --dynamixel-port /dev/ttyACM0 --pan-sign 1 --tilt-sign -1 \
  --scrfd-hef "$PWD/models/hailo8/scrfd_10g.hef" \
  --personface-hef "$PWD/models/hailo8/yolov8m.hef"
```

```bash
python3 vital_run.py --camera 0 \
  --dynamixel-port /dev/ttyACM0 --pan-sign 1 --tilt-sign -1 \
  --scrfd-hef "$PWD/models/hailo8/scrfd_10g.hef" \
  --personface-hef "$PWD/models/hailo8/yolov8m.hef"
```

Adaptive respiration capture (repeat for `rr_adaptive_torso.npz`, paced at 10,
15, and 20 brpm; record first LOCK, retention, sources, accepted baseline
count, and FPS):

```bash
python3 vital_run.py --camera 0 \
  --dynamixel-port /dev/ttyACM0 --pan-sign 1 --tilt-sign -1 \
  --scrfd-hef "$PWD/models/hailo8/scrfd_10g.hef" \
  --personface-hef "$PWD/models/hailo8/yolov8m.hef" \
  --dump sessions/rr_adaptive_face_only.npz
```

Pi Hailo acceptance:

```bash
BIO_HAILO_HARDWARE=1 \
BIO_HAILO_SCRFD_HEF="$PWD/models/hailo8/scrfd_10g.hef" \
BIO_HAILO_PERSONFACE_HEF="$PWD/models/hailo8/yolov8m.hef" \
python3 -m unittest tests.test_hailo_hardware -v
```

Hailo 문제를 분리 진단할 때만 명시적으로 `--face-detector rppg`를 사용한다. 이것은 대회용 hybrid의 성능/복구 검증을 대신하는 운영 폴백이 아니다.

```bash
python3 vital_monitor.py --camera 0 --face-detector rppg \
  --dynamixel-port /dev/ttyACM0 --pan-sign 1 --tilt-sign -1
```

## 7. 대회 acceptance matrix

각 행에서 한 사람만 화면에 두고 `source`, `result_age`, 첫 관측/복구 latency, application/wall FPS, 모델 completion 증가, 실제 wrong-direction command 횟수를 기록한다.

| 장면 | 기대 source/동작 | 합격 조건 |
|---|---|---|
| 중앙 | `SCRFD` | 신선한 `sensor_box`와 `motor_box`, 안정된 중앙 유지 |
| 좌/우/상/하 가장자리 | `SCRFD_EDGE` 후 `SCRFD` 복귀 가능 | 네 가장자리 모두 **같은 사용자**를 중앙으로 복귀; edge box는 센서 ROI 금지 |
| yaw/pitch | SCRFD 또는 일시적 복구 source | 방향이 바뀌어도 같은 사용자 복구, wrong-direction 0 |
| 빠른 이동 | `FLOW`/`PERSON_HEAD` 후 `SCRFD` | recovery latency 기록, 새 사용자를 목표로 삼지 않음 |
| 역광 | SCRFD 또는 보수적 복구 | 불확실하면 센서 ROI 없음; 중앙 복귀 뒤 SCRFD 재획득 |
| 빈 프레임 | `NONE` | 새 motor goal 0, 이전 센서 ROI 재사용 금지 |

모든 edge가 같은 사용자를 재중앙화해야 최종 합격이다. `FLOW`, `PERSON_HEAD`, `SCRFD_EDGE`는 절대로 센서 ROI가 될 수 없다. 불확실/empty 상태는 새 goal을 만들면 안 된다.

## 8. 문제 해결

- **HAILO8L mismatch**: 즉시 중단한다. `identify`, probe의 절대 HEF 경로, downloader `--arch hailo8`을 다시 확인한다. HAILO8L HEF rename은 해결책이 아니다.
- **YOLO label 계약**: 공식 `yolov8m.hef`는 내장 COCO class 0=`person` 계약을 사용하므로 `label_path=None`이 정상이다. custom `yolov8n_personface`만 sidecar의 class 0=`person`, class 1=`face`가 필수다.
- **SCRFD 9 output/landmark tensor**: SCRFD는 score 3개, bbox 3개, landmark 3개가 나올 수 있다. decoder는 landmark를 bbox로 고르지 않아야 한다. output vstream name/shape를 probe와 실패 메시지에서 보존한다.
- **callback output shape 오류**: 두 HEF가 기대 postprocess 계약인지, input/output vstream shape/type, Hailo Apps helper release를 확인한다. 임의 reshape 대신 probe의 full metadata와 정확한 모델 경로를 기록한다.
- **warmup 뒤 completion 0**: `hailortcli scan`, driver log(`dmesg | grep -i hailo`), Hailo Apps environment, HEF architecture를 확인한다. callback error와 deferred runner close도 함께 본다.
- **Picamera2/OpenCV 실패**: `rpicam-hello`로 libcamera를 먼저 검증하고 `python3 -c "from picamera2 import Picamera2"`, `python3 -c "import cv2; print(cv2.__version__)"`를 실행한다. probe는 두 실패 원인을 모두 출력한다.
- **headless/DISPLAY**: probe에는 창이 없다. production GUI가 필요 없으면 `vital_monitor.py --headless`를 사용한다. `vital_run.py` Tk GUI는 유효한 `DISPLAY`가 필요하다.
- **motor torque/watchdog**: probe와 hardware unittest에서는 motor가 전혀 초기화되지 않는다. production 전 self-test로 `pan-sign`/`tilt-sign`, torque enable, 통신 baud/port, watchdog/종료 시 torque-off를 확인한다. wrong-direction이면 즉시 중단한다.
- **자동 model search가 다른 파일을 선택**: production 명령에 위와 같이 `--scrfd-hef`와 `--personface-hef` 절대 경로를 다시 명시한다. probe가 출력한 두 경로가 검증한 파일과 byte-for-byte 같은지 확인한다.
- **firmware/driver/runtime 불일치**: package 하나만 바꾸지 않는다. `dpkg-query`, `hailortcli --version`, `identify` 결과를 함께 보존하고 Hailo의 4.23 호환 세트로 복구한다.
