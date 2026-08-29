from dynamixel_sdk import PortHandler, PacketHandler

# 통신 및 모터 설정값
DEVICE_NAME = '/dev/ttyACM0'
PROTOCOL_VERSION = 2.0
BAUDRATE = 57600
DXL_ID = 1

for DXL_ID in (1, 2):  # 팬/틸트 모터 ID
    # 포트 및 패킷 핸들러 초기화
    portHandler = PortHandler(DEVICE_NAME)
    packetHandler = PacketHandler(PROTOCOL_VERSION)

    # 포트 열기 및 통신 속도 설정
    portHandler.openPort()
    portHandler.setBaudRate(BAUDRATE)

    # 토크 해제 명령 전송
    ADDR_TORQUE_ENABLE = 64
    TORQUE_DISABLE = 0
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)

    # 포트 닫기
    portHandler.closePort()