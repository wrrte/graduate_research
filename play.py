import gymnasium as gym
import ale_py
import numpy as np
import cv2
import datetime
import os
import sys
import keyboard

Game = "Seaquest"  # 게임 이름을 상수로 정의

# Gymnasium 1.0.0 이상 버전에서는 외부 환경(ALE)을 명시적으로 등록해야 합니다.
gym.register_envs(ale_py)

class Atari(gym.Env):
    LOCK = None
    metadata = {}

    def __init__(
        self,
        name,
        action_repeat=4,
        size=(64, 64),
        gray=False,
        noops=0,
        lives="discount",
        sticky=False,
        actions="needed",
        length=108000,
        resize="opencv",
        seed=None,
    ):
        assert size[0] == size[1]
        assert lives in ("unused", "discount", "reset"), lives
        assert actions in ("all", "needed"), actions
        assert resize in ("opencv", "pillow"), resize
        if self.LOCK is None:
            import multiprocessing as mp

            mp = mp.get_context("spawn")
            self.LOCK = mp.Lock()
        self._resize = resize
        if self._resize == "opencv":
            import cv2
            self._cv2 = cv2
        if self._resize == "pillow":
            from PIL import Image
            self._image = Image

        self._env_id = name
        self._repeat = action_repeat
        self._size = size
        self._gray = gray
        self._noops = noops
        self._lives = lives
        self._sticky = sticky
        self._length = length
        self._random = np.random.RandomState(seed)
        with self.LOCK:
            self._env = gym.make(
                id=self._env_id,
                obs_type="rgb" if not gray else "grayscale",
                frameskip=1,
                repeat_action_probability=0.25 if sticky else 0.0,
                full_action_space=(actions == "all"),
            )
        assert self._env.unwrapped.get_action_meanings()[0] == "NOOP"
        self._seed = seed
        self._is_init = True
        shape = self._env.observation_space.shape
        self._buffer = [np.zeros(shape, np.uint8) for _ in range(2)]
        self._ale = self._env.unwrapped.ale
        self._last_lives = None
        self._done = True
        self._step = 0
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        img_shape = self._size + ((1,) if self._gray else (3,))
        return gym.spaces.Dict({"image": gym.spaces.Box(0, 255, img_shape, np.uint8)})

    @property
    def action_space(self):
        space = self._env.action_space
        space.discrete = True
        return space

    def step(self, action):
        total = 0.0
        dead = False
        for repeat in range(self._repeat):
            _, reward, terminated, truncated, info = self._env.step(action)
            self._step += 1
            total += reward
            if repeat == self._repeat - 2:
                self._screen(self._buffer[1])
            if terminated or truncated:
                break
            if self._lives != "unused":
                current = self._ale.lives()
                if current < self._last_lives:
                    dead = True
                    self._last_lives = current
                    break
        if not self._repeat:
            self._buffer[1][:] = self._buffer[0][:]
        self._screen(self._buffer[0])
        my_truncated = self._length and self._step >= self._length
        self._done = terminated or my_truncated or truncated
        return self._obs(
            total,
            info,
            is_last=self._done or (dead and self._lives == "reset"),
            is_terminal=dead or terminated,
        )

    def reset(self):
        _, info = self._env.reset()
        if self._noops:
            for _ in range(self._random.randint(self._noops)):
                _, _, dead, _ = self._env.step(0)
                if dead: self._env.reset()
        self._last_lives = self._ale.lives()
        self._screen(self._buffer[0])
        self._buffer[1].fill(0)
        self._done = False
        self._step = 0
        obs, reward, is_last, info = self._obs(0.0, info, is_first=True)
        return obs, info

    def _obs(self, reward, info, is_first=False, is_last=False, is_terminal=False):
        # 화면 깜빡임 방지를 위해 최근 2프레임의 최대값을 취합니다.
        np.maximum(self._buffer[0], self._buffer[1], out=self._buffer[0])
        
        # 64x64로 리사이즈하기 전의 원본 해상도(160x210)를 백업하여 info에 저장합니다.
        original_image = self._buffer[0].copy()
        info['original_screen'] = original_image 
        
        image = self._buffer[0]
        if image.shape[:2] != self._size:
            if self._resize == "opencv":
                image = self._cv2.resize(image, self._size, interpolation=self._cv2.INTER_AREA)
            if self._resize == "pillow":
                image = self._image.fromarray(image)
                image = image.resize(self._size, self._image.NEAREST)
                image = np.array(image)
        if self._gray:
            weights = [0.299, 0.587, 1 - (0.299 + 0.587)]
            image = np.tensordot(image, weights, (-1, 0)).astype(image.dtype)
            image = image[:, :, None]
        info['is_first'] = is_first
        info['is_terminal'] = is_terminal
        if 'episode_frame_number' in info:
            info['episode_frame_number'] //= self._repeat
            
        return image, reward, is_last, info

    def _screen(self, array):
        self._ale.getScreenRGB(array)

    def close(self):
        return self._env.close()

def save_episode(data_list):
    """수집된 에피소드 데이터를 .npz 파일로 저장합니다."""
    if not data_list: return
    
    os.makedirs('demonstrations', exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"demonstrations/{Game}_{timestamp}.npz"
    
    # 리스트를 넘파이 배열로 변환
    observations = np.array([d[0] for d in data_list], dtype=np.uint8)
    actions = np.array([d[1] for d in data_list], dtype=np.int32)
    rewards = np.array([d[2] for d in data_list], dtype=np.float32)
    dones = np.array([d[3] for d in data_list], dtype=bool)
    
    np.savez_compressed(filename, obs=observations, action=actions, reward=rewards, done=dones)
    print(f"\n[저장 완료] 에피소드가 {filename}에 저장되었습니다. (길이: {len(data_list)})")

def copy_image_to_clipboard(filepath):
    """지정된 경로의 이미지를 클립보드에 복사합니다 (Windows 전용, 비동기 실행)."""
    if os.name == 'nt':
        import subprocess
        import threading
        def _copy():
            abs_path = os.path.abspath(filepath)
            cmd = [
                "powershell",
                "-command",
                f"Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Clipboard]::SetImage([System.Drawing.Image]::FromFile('{abs_path}'))"
            ]
            # 새 콘솔 창이 뜨지 않도록 설정 플래그 사용 (CREATE_NO_WINDOW)
            subprocess.run(cmd, creationflags=0x08000000)
        threading.Thread(target=_copy).start()

def manual_play_with_save(mode="rl"):
    if os.name == 'posix' and sys.platform != 'darwin' and not os.environ.get('DISPLAY'):
        print("\n[오류] 디스플레이(모니터)를 찾을 수 없습니다.")
        print("서버 환경(Headless)에서는 게임 화면을 띄울 수 없습니다.")
        print("이 스크립트를 로컬 PC에서 실행한 후, 생성된 .npz 파일을 서버로 복사해 주세요.\n")
        return

    length = 0 if mode == "ppt" else 108000
    env = Atari(name=f'ALE/{Game}-v5', action_repeat=4, size=(64, 64), gray=False, length=length)
    obs, info = env.reset()
    
    total_reward = 0
    episode_data = [] 
    
    print("\n" + "="*30)
    if mode == "ppt":
        print(f"{Game} 직접 플레이 & PPT 발표 모드 (시간 제한 없음)")
    else:
        print(f"{Game} 직접 플레이 & 데이터 저장 모드")
    print("조작: WASD(이동), Space(Action/Fire), Enter(일시정지), P(스크린샷), Q(저장 후 종료)")
    print("="*30)

    try:
        while True:
            # 1. 화면 렌더링 (info에 저장해 둔 고해상도 원본 이미지를 활용)
            hi_res_img = info.get('original_screen', obs)
            raw_bgr_img = cv2.cvtColor(hi_res_img, cv2.COLOR_RGB2BGR)
            
            # 게임 플레이 화면용 (전체 여백 유지: 160x210 원본 -> 480x630)
            display_img = cv2.resize(raw_bgr_img, (480, 630), interpolation=cv2.INTER_LINEAR)
            
            # [크롭] 스크린샷용 깔끔한 이미지 생성
            # y축(세로): 상단의 검은 여백(약 26px)과 하단 여백 및 로고(약 176px 이후) 전면 제거
            # x축(가로): 좌우의 검은 여백(각 약 8px) 제거
            cropped_bgr_img = raw_bgr_img[26:183, 8:152]
            display_img_clean = cv2.resize(cropped_bgr_img, (480, 500), interpolation=cv2.INTER_LINEAR)
            
            cv2.putText(display_img, f"Score: {int(total_reward)}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            # 창 이름을 Game으로 변경
            cv2.imshow(f"{Game} - Manual Play & Record", display_img)
            
            # 2. 다중 키 입력 처리 (keyboard 라이브러리 활용)
            cv2.waitKey(50) # 화면 렌더링 유지 및 프레임 속도 조절
            
            action = 0
            w = keyboard.is_pressed('w')
            a = keyboard.is_pressed('a')
            s = keyboard.is_pressed('s')
            d = keyboard.is_pressed('d')
            space = keyboard.is_pressed('space')
            
            if keyboard.is_pressed('q'): 
                save_episode(episode_data)
                break
                
            # 스크린샷 (P 키)
            if keyboard.is_pressed('p'):
                os.makedirs('screenshots', exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                screenshot_path = f"screenshots/screenshot_{timestamp}.png"
                cv2.imwrite(screenshot_path, display_img_clean)
                copy_image_to_clipboard(screenshot_path)
                print(f"[스크린샷 저장 및 클립보드 복사 완료] {screenshot_path}")
                cv2.waitKey(200) # 중복 캡처 방지
                
            # 일시 정지 (Enter 키)
            if keyboard.is_pressed('enter'):
                cv2.putText(display_img, "PAUSED", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow(f"{Game} - Manual Play & Record", display_img)
                cv2.waitKey(300) # 중복 입력 방지 (Debounce)
                while True:
                    if keyboard.is_pressed('enter'):
                        cv2.waitKey(300)
                        break
                    cv2.waitKey(50)
                
            # 복합 키(동시 입력) 매핑
            if w and d and space: action = 14
            elif w and a and space: action = 15
            elif s and d and space: action = 16
            elif s and a and space: action = 17
            elif w and space: action = 10
            elif d and space: action = 11
            elif a and space: action = 12
            elif s and space: action = 13
            elif w and d: action = 6
            elif w and a: action = 7
            elif s and d: action = 8
            elif s and a: action = 9
            elif w: action = 2
            elif s: action = 5
            elif a: action = 4
            elif d: action = 3
            elif space: action = 1
            
            # 3. 데이터 저장 (모델 규격에 맞는 64x64 축소 이미지인 obs를 저장)
            # 4. 환경 진행
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            
            episode_data.append((obs, action, reward, done))
            obs = next_obs
            
            if done:
                print(f"Game Over! 최종 점수: {total_reward}")
                save_episode(episode_data)
                obs, info = env.reset()
                total_reward = 0
                episode_data = [] 
                
    finally:
        env.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Private Eye Manual Play")
    parser.add_argument('--mode', type=str, default='rl', choices=['rl', 'ppt'], 
                        help="Mode 'rl' limits length to 108000. Mode 'ppt' has no time limit.")
    args = parser.parse_args()
    
    manual_play_with_save(mode=args.mode)