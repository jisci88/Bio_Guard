"""
csv_uploader.py - 측정 CSV 를 원격 서버로 실시간 전송한다.

설계 원칙 하나: **계측을 절대 막지 않는다.**

  put() 은 행을 메모리 버퍼에 넣고 즉시 반환한다. 실제 전송은 이 스레드가
  따로 하고, 실패하면 버퍼에 남겨 두었다가 다음 주기에 다시 보낸다.
  네트워크가 끊기든 서버가 죽든 HR/RR/체온 측정은 그대로 돌아간다.
  로컬 CSV 가 원본이고 전송은 사본이므로, 전송이 실패해도 데이터는 안 잃는다.

전송 방식은 ssh CLI 파이프다. 라즈베리파이에 기본 설치되어 있어 pip 설치가
필요 없고, 한 주기에 ssh 프로세스를 딱 하나만 띄운다:

    ssh -i KEY user@host "mkdir -p DIR && ([ -s F ] || printf HEADER > F) && cat >> F"

새 행들을 그 표준입력으로 흘려보낸다. 원격 파일은 append 되므로 세션 중간에
연결이 끊겼다 붙어도 이어서 쌓인다.

연결 확인 (카메라 없이):

    python3 csv_uploader.py --test
"""

import csv
import io
import os
import shlex
import subprocess
import threading
import time
from collections import deque

import config
from config import get_logger

log = get_logger("upload")


def resolve_key(path):
    """키 파일을 찾고, POSIX 라면 권한을 600 으로 맞춘다.

    ssh 는 권한이 열린 개인키를 거부한다("UNPROTECTED PRIVATE KEY FILE").
    윈도우에서 복사해 온 키가 644 로 올라오는 일이 흔해서 여기서 바로잡는다.
    """
    if not path:
        return None

    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (path, os.path.join(here, path)):
        if os.path.isfile(candidate):
            key = os.path.abspath(candidate)
            if os.name == "posix":
                try:
                    if os.stat(key).st_mode & 0o077:
                        os.chmod(key, 0o600)
                        log.info("키 파일 권한을 600 으로 조정했습니다: %s", key)
                except OSError as exc:
                    log.warning("키 파일 권한 조정 실패: %s", exc)
            return key

    log.warning("키 파일을 찾을 수 없습니다: %s", path)
    return None


def encode_row(values):
    """csv 모듈로 한 행을 문자열로 만든다 (따옴표 처리를 로컬 파일과 동일하게)."""
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerow(values)
    return buf.getvalue()


class CsvUploader(threading.Thread):
    """CSV 행을 모아 주기적으로 원격 서버에 append 한다."""

    MAX_BUFFER = 5000        # 이 이상 밀리면 오래된 행부터 버린다

    def __init__(self, host=None, key_path=None, remote_path=None,
                 header=None, user=None, port=None, interval=None,
                 timeout=None):
        super().__init__(daemon=True)
        self.host = host or config.REMOTE_HOST
        self.port = int(port or config.REMOTE_PORT)
        self.key = resolve_key(key_path or config.REMOTE_KEY)
        self.interval = float(interval or config.REMOTE_INTERVAL)
        self.timeout = float(timeout or config.REMOTE_TIMEOUT)

        remote_path = remote_path or "vitals/session.csv"
        self.remote_path = remote_path
        self.remote_dir = os.path.dirname(remote_path) or "."
        self.header_text = encode_row(header) if header else None

        # 계정을 직접 주면 탐색을 건너뛴다. 그리고 전송이 실패해도 그 지정을
        # 버리지 않는다 - 사용자가 정한 계정을 마음대로 바꿔 시도하면 안 된다.
        self.user = user
        self._explicit_user = user is not None
        self.user_candidates = ([user] if user
                                else list(config.REMOTE_USERS))

        self._buffer = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()

        self.sent = 0
        self.dropped = 0
        self.failures = 0
        self.last_error = ""
        self.connected = False

    # ------------------------------------------------------------ 공개 API

    def put(self, values):
        """행 하나를 전송 대기열에 넣는다. 논블로킹이며 절대 예외를 내지 않는다."""
        try:
            row = encode_row(values)
        except Exception as exc:          # 계측을 막을 이유가 되지 않는다
            log.debug("행 인코딩 실패: %s", exc)
            return
        with self._lock:
            self._buffer.append(row)
            while len(self._buffer) > self.MAX_BUFFER:
                self._buffer.popleft()
                self.dropped += 1

    def stop(self, flush_timeout=None):
        """남은 행을 한 번 더 보내고 스레드를 정리한다."""
        self._stop.set()
        self.join(timeout=flush_timeout if flush_timeout is not None
                  else self.timeout + 2.0)
        with self._lock:
            pending = len(self._buffer)
        if pending:
            log.warning("전송하지 못한 행 %d개가 남았습니다 (로컬 CSV 에는 있음).",
                        pending)
        log.info("원격 전송 종료: 성공 %d행, 유실 %d행, 실패 %d회",
                 self.sent, self.dropped, self.failures)

    def status(self):
        with self._lock:
            pending = len(self._buffer)
        state = "OK" if self.connected else "OFF"
        return (f"[NET] {state} {self.host} sent={self.sent} "
                f"queue={pending} fail={self.failures}")

    # -------------------------------------------------------------- 내부

    def _ssh_base(self):
        cmd = ["ssh", "-p", str(self.port),
               "-o", "BatchMode=yes",                     # 비밀번호 프롬프트 금지
               "-o", "StrictHostKeyChecking=accept-new",  # 첫 접속 자동 수락
               "-o", f"ConnectTimeout={int(self.timeout)}"]
        if self.key:
            cmd += ["-i", self.key]
        return cmd

    def _remote_command(self):
        directory = shlex.quote(self.remote_dir)
        target = shlex.quote(self.remote_path)
        parts = [f"mkdir -p {directory}"]
        if self.header_text:
            # 파일이 비어 있을 때만 헤더를 쓴다. 재접속해도 중복되지 않는다.
            parts.append(
                f"([ -s {target} ] || printf '%s' {shlex.quote(self.header_text)} "
                f"> {target})"
            )
        parts.append(f"cat >> {target}")
        return " && ".join(parts)

    def _run_ssh(self, user, remote_cmd, payload=None):
        cmd = self._ssh_base() + [f"{user}@{self.host}", remote_cmd]
        try:
            return subprocess.run(cmd, input=payload, capture_output=True,
                                  timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self._note("전송 시간 초과")
        except FileNotFoundError:
            self._note("ssh 명령을 찾을 수 없습니다 (openssh-client 설치 필요)")
        except OSError as exc:
            self._note(f"ssh 실행 실패: {exc}")
        except Exception as exc:
            # 어떤 예외도 이 스레드를 죽이면 안 된다. 죽으면 전송이 조용히
            # 멈추고, 로그를 볼 때까지 아무도 모른다.
            self._note(f"예상치 못한 전송 오류: {type(exc).__name__}: {exc}")
        return None

    def _note(self, message):
        self.failures += 1
        self.last_error = str(message)
        self.connected = False
        # 매번 찍으면 로그가 도배된다. 처음과 그 뒤 10회마다.
        if self.failures == 1 or self.failures % 10 == 0:
            log.warning("원격 전송 실패 %d회: %s", self.failures, self.last_error)

    def _resolve_user(self):
        """오라클 이미지마다 기본 계정이 달라 순서대로 접속을 시도한다."""
        if self.user:
            return self.user
        for candidate in self.user_candidates:
            done = self._run_ssh(candidate, "true")
            if done is not None and done.returncode == 0:
                self.user = candidate
                log.info("원격 접속 성공: %s@%s:%s -> %s",
                         candidate, self.host, self.port, self.remote_path)
                return candidate
        self._note(f"접속 가능한 계정을 찾지 못했습니다 (시도: {self.user_candidates})")
        return None

    def _flush(self):
        with self._lock:
            if not self._buffer:
                return
            rows = [self._buffer.popleft() for _ in range(len(self._buffer))]

        failed = True
        try:
            user = self._resolve_user()
            if user is None:
                return
            done = self._run_ssh(user, self._remote_command(),
                                 payload="".join(rows).encode("utf-8"))
            if done is None:
                return
            if done.returncode != 0:
                self._note(done.stderr.decode("utf-8", "replace").strip()[:200]
                           or f"ssh 종료코드 {done.returncode}")
                if not self._explicit_user:
                    # 자동 탐색으로 정한 계정이라면 다음 주기에 다시 찾는다.
                    self.user = None
                return

            if not self.connected:
                log.info("원격 전송 시작: %s@%s:%s", user, self.host,
                         self.remote_path)
            self.connected = True
            self.sent += len(rows)
            self.failures = 0
            self.last_error = ""
            failed = False
        finally:
            if failed:
                # 못 보낸 행은 되돌려 놓고 다음 주기에 재시도한다.
                with self._lock:
                    self._buffer.extendleft(reversed(rows))
                    while len(self._buffer) > self.MAX_BUFFER:
                        self._buffer.popleft()
                        self.dropped += 1

    def run(self):
        # _flush 안에서 무슨 일이 나든 스레드는 살아 있어야 한다.
        while not self._stop.wait(self.interval):
            self._safe_flush()
        self._safe_flush()     # 종료 직전 마지막 한 번

    def _safe_flush(self):
        try:
            self._flush()
        except Exception as exc:
            self._note(f"전송 루프 오류: {type(exc).__name__}: {exc}")


# ══════════════════════════════════════════════════════
#  연결 확인
# ══════════════════════════════════════════════════════

def test_connection(host=None, key_path=None, user=None, port=None,
                    remote_dir=None):
    """카메라 없이 접속과 쓰기 권한만 확인한다."""
    host = host or config.REMOTE_HOST
    remote_dir = remote_dir or config.REMOTE_DIR
    remote_path = f"{remote_dir}/_connection_test.csv"

    uploader = CsvUploader(host=host, key_path=key_path, user=user, port=port,
                           remote_path=remote_path,
                           header=["time", "note"])

    print("=" * 60)
    print(f"  원격 전송 점검  {host}:{uploader.port}")
    print(f"  키   : {uploader.key or '(없음 - ssh 기본 키 사용)'}")
    print(f"  계정 : {uploader.user or ' / '.join(uploader.user_candidates)}")
    print(f"  경로 : {remote_path}")
    print("=" * 60)

    if uploader.key is None and key_path:
        print("\n>> 키 파일을 찾지 못했습니다. --key 경로를 확인하세요.")
        return False

    uploader.put([time.strftime("%Y-%m-%d %H:%M:%S"), "connection test"])
    uploader._flush()

    if uploader.connected:
        print(f"\n>> 성공: {uploader.user}@{host} 로 {uploader.sent}행 전송")
        print(f"   서버에서 확인:  cat {remote_path}")
        return True

    print(f"\n>> 실패: {uploader.last_error}")
    print("   확인할 것:")
    print("   - 오라클 클라우드 보안목록/방화벽에서 22번 포트가 열려 있는지")
    print("   - 키 파일이 이 서버에 등록된 키가 맞는지")
    print(f"   - 계정 이름 (자동 탐색: {uploader.user_candidates})")
    print("     다르면 --user 로 직접 지정하세요")
    return False


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="측정 CSV 원격 전송")
    ap.add_argument("--test", action="store_true", help="접속과 쓰기 권한만 확인")
    ap.add_argument("--host", default=config.REMOTE_HOST)
    ap.add_argument("--port", type=int, default=config.REMOTE_PORT)
    ap.add_argument("--key", default=config.REMOTE_KEY)
    ap.add_argument("--user", default=None,
                    help=f"미지정 시 자동 탐색: {', '.join(config.REMOTE_USERS)}")
    ap.add_argument("--dir", default=config.REMOTE_DIR)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    config.setup_logging(level=args.log_level)

    if not args.test:
        ap.print_help()
        raise SystemExit

    ok = test_connection(host=args.host, key_path=args.key, user=args.user,
                         port=args.port, remote_dir=args.dir)
    raise SystemExit(0 if ok else 1)
