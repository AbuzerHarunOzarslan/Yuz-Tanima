import cv2, mediapipe as mp, numpy as np, time, pygame, sys, os, urllib.request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

SCREEN_W, SCREEN_H = 1280, 720
CAM_W,    CAM_H    = 640,  480
FPS                = 30

EAR_BLINK     = 0.21
WINK_DIFF     = 0.055
EAR_SMOOTH    = 0.28
MOVE_COOLDOWN = 0.55
BLINK_WIN     = 1.0

LETTERS = [
    'A','B','C','Ç','D','E','F','G','Ğ',
    'H','I','İ','J','K','L','M','N','O',
    'Ö','P','R','S','Ş','T','U','Ü','V',
    'Y','Z','⌫','␣','✓'
]
COLS = 9
ROWS = (len(LETTERS) + COLS - 1) // COLS
def rlen(r): return min(COLS, len(LETTERS) - r*COLS)

BG       = (12, 14, 22)
C_NORM   = (26, 30, 48)
C_ROW    = (34, 40, 68)
C_CUR    = (22, 95, 230)
C_FLASH  = (0, 210, 110)
C_SPEC   = (68, 34, 128)
B_NORM   = (44, 52, 80)
B_HL     = (22, 95, 230)
TXT      = (220, 228, 255)
TXT_D    = (90, 100, 138)
TXT_BLK  = (12, 14, 22)
ACCENT   = (22, 95, 230)
SENT_BG  = (18, 20, 34)
GREEN    = (0, 200, 80)
RED      = (220, 55, 55)
YELLOW   = (255, 205, 0)
ORANGE   = (255, 140, 0)

pygame.init()
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("Felcli Hasta Iletisim v3")
clock = pygame.time.Clock()

def fnt(size, bold=False):
    for n in ["calibri","segoeui","arial"]:
        try: return pygame.font.SysFont(n, size, bold=bold)
        except: pass
    return pygame.font.Font(None, size)

F_XL = fnt(48, True);  F_LG = fnt(38, True)
F_MD = fnt(26, True);  F_SM = fnt(19)
F_XS = fnt(15)

GL=20; GT=168; CW=108; CH=88; GAP=8
CAM_X = GL + COLS*(CW+GAP) + 18
CAM_Y = GT
CDW   = SCREEN_W - CAM_X - 12
CDH   = int(CDW * CAM_H / CAM_W)

def crect(r, c): return (GL+c*(CW+GAP), GT+r*(CH+GAP), CW, CH)

MDIR  = os.path.join(os.environ.get("LOCALAPPDATA","C:/Temp"), "EyeComm")
os.makedirs(MDIR, exist_ok=True)
MPATH = os.path.join(MDIR, "face_landmarker.task")
MURL  = ("https://storage.googleapis.com/mediapipe-models/"
         "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

def download_model():
    if not os.path.exists(MPATH):
        print("Model indiriliyor..."); urllib.request.urlretrieve(MURL, MPATH); print("Hazir.")

L_EAR = [362,385,387,263,373,380]
R_EAR = [33, 160,158,133,153,144]

def ear(lm, idx, W, H):
    p = np.array([[lm[i].x*W, lm[i].y*H] for i in idx])
    A=np.linalg.norm(p[1]-p[5]); B=np.linalg.norm(p[2]-p[4]); C=np.linalg.norm(p[0]-p[3])
    return (A+B)/(2.0*C+1e-6)

class State:
    def __init__(self):
        self.sentence    = ""
        self.row = self.col = 0
        self.flash_rc    = None
        self.flash_until = 0.0
        self.l_s = self.r_s = 0.30
        self.eye_state   = "OPEN"
        self.blink_count = 0
        self.blink_win   = None
        self.last_move   = 0.0
        self.face_ok     = False
        self.status      = "Baslatiliyor..."

    @property
    def letter(self):
        idx = self.row*COLS + self.col
        return LETTERS[idx] if idx < len(LETTERS) else ""

    def clamp(self):
        self.row = max(0, min(ROWS-1, self.row))
        self.col = max(0, min(rlen(self.row)-1, self.col))

    def right(self):
        if self.col < rlen(self.row)-1: self.col += 1
        self.status = f">> {self.letter}"

    def left(self):
        if self.col > 0: self.col -= 1
        self.status = f"<< {self.letter}"

    def down(self):
        if self.row < ROWS-1: self.row += 1; self.clamp()
        self.status = f"Asagi — Satir {self.row+1}"

    def up(self):
        if self.row > 0: self.row -= 1; self.clamp()
        self.status = f"Yukari — Satir {self.row+1}"

    def select(self):
        l = self.letter
        if not l: return
        if   l=='⌫': self.sentence = self.sentence[:-1]
        elif l=='␣': self.sentence += ' '
        elif l=='✓': pass
        else:        self.sentence += l
        self.flash_rc    = (self.row, self.col)
        self.flash_until = time.time() + 0.45
        self.status      = f"'{l}' eklendi"

    def on_blinks(self, n):
        if   n==1: self.select()
        elif n==2: self.down()
        elif n>=3: self.up()

    def smooth_ear(self, l_raw, r_raw):
        a = EAR_SMOOTH
        self.l_s = a*l_raw + (1-a)*self.l_s
        self.r_s = a*r_raw + (1-a)*self.r_s

    def classify(self):
        l, r   = self.l_s, self.r_s
        diff   = l - r
        l_cl   = l < EAR_BLINK
        r_cl   = r < EAR_BLINK

        if not l_cl and not r_cl:
            return "OPEN"

        if l_cl and r_cl:
            if   diff  >  WINK_DIFF: return "R_WINK"
            elif diff  < -WINK_DIFF: return "L_WINK"
            else:                    return "BOTH"

        if r_cl and diff >  WINK_DIFF: return "R_WINK"
        if l_cl and diff < -WINK_DIFF: return "L_WINK"
        return "OPEN"

st = State()

def rrect(surf, col, rect, r=10):
    x,y,w,h = rect
    pygame.draw.rect(surf, col, (x+r,y,w-2*r,h))
    pygame.draw.rect(surf, col, (x,y+r,w,h-2*r))
    for cx,cy in [(x+r,y+r),(x+w-r,y+r),(x+r,y+h-r),(x+w-r,y+h-r)]:
        pygame.draw.circle(surf, col, (cx,cy), r)

def txtc(surf, text, f, col, rect):
    x,y,w,h = rect
    s = f.render(text, True, col)
    surf.blit(s, (x+(w-s.get_width())//2, y+(h-s.get_height())//2))

def hbar(x,y,val,maxv,w,h,col,lbl):
    pygame.draw.rect(screen,(28,33,52),(x,y,w,h),border_radius=3)
    fw=int(min(val/maxv,1.0)*w)
    if fw>0: pygame.draw.rect(screen,col,(x,y,fw,h),border_radius=3)
    screen.blit(F_XS.render(lbl,True,TXT_D),(x,y+h+3))

STATE_INFO = {
    "OPEN":   ("Acik",         GREEN),
    "R_WINK": ("SAG WINK >>",  YELLOW),
    "L_WINK": ("<< SOL WINK",  YELLOW),
    "BOTH":   ("CIFT KIRPMA",  ORANGE),
}

def draw(csf):
    now = time.time()
    screen.fill(BG)

    screen.blit(F_XL.render("Goz Kontrollu Iletisim  v3", True, TXT), (GL, 8))

    fc = GREEN if st.face_ok else RED
    ft = "YUZ ALGILANDI" if st.face_ok else "! YUZ ALGILANAMADI — YAKIN DURUN"
    screen.blit(F_SM.render(ft, True, fc), (GL, 64))

    bx,by,bw,bh = GL, 90, COLS*(CW+GAP)-GAP, 60
    rrect(screen, SENT_BG, (bx,by,bw,bh), 10)
    pygame.draw.rect(screen, B_NORM, (bx,by,bw,bh), 1, border_radius=10)
    disp = st.sentence[-55:] if st.sentence else "Harf seciniz..."
    ss   = F_MD.render(disp, True, TXT if st.sentence else TXT_D)
    screen.blit(ss, (bx+12, by+16))
    if st.sentence and int(now*2)%2==0:
        pygame.draw.rect(screen, ACCENT, (bx+12+ss.get_width()+3, by+13, 3, 34))

    for row in range(ROWS):
        ar = (row == st.row)
        for col in range(rlen(row)):
            idx = row*COLS+col
            ltr = LETTERS[idx]
            x,y,w,h = crect(row, col)
            is_c  = ar and col==st.col
            is_fl = st.flash_rc==(row,col) and now<st.flash_until
            is_sp = ltr in ('⌫','␣','✓')

            if   is_fl: bg=C_FLASH
            elif is_c:  bg=C_CUR
            elif ar:    bg=C_ROW
            elif is_sp: bg=C_SPEC
            else:       bg=C_NORM
            rrect(screen, bg, (x,y,w,h), 10)

            brd  = B_HL if (is_c or ar) else B_NORM
            brdw = 3 if is_c else (2 if ar else 1)
            pygame.draw.rect(screen, brd, (x,y,w,h), brdw, border_radius=10)
            txtc(screen, ltr, F_LG if is_c else F_MD,
                 TXT_BLK if is_fl else TXT, (x,y,w,h))

    if csf:
        sc = pygame.transform.scale(csf, (CDW, CDH))
        screen.blit(sc, (CAM_X, CAM_Y))
        pygame.draw.rect(screen, B_NORM, (CAM_X,CAM_Y,CDW,CDH), 2, border_radius=6)

        ey = CAM_Y + CDH + 8
        hbar(CAM_X,     ey, st.l_s, 0.35, 95, 11,
             RED if st.l_s<EAR_BLINK else GREEN, f"Sol EAR {st.l_s:.3f}")
        hbar(CAM_X+110, ey, st.r_s, 0.35, 95, 11,
             RED if st.r_s<EAR_BLINK else GREEN, f"Sag EAR {st.r_s:.3f}")

        diff = st.l_s - st.r_s
        screen.blit(F_XS.render(f"Asimetri: {diff:+.3f}  (esik: ±{WINK_DIFF:.3f})",
                                 True, TXT_D), (CAM_X, ey+26))

        info = STATE_INFO.get(st.eye_state, (st.eye_state, TXT_D))
        screen.blit(F_MD.render(f"Goz: {info[0]}", True, info[1]), (CAM_X, ey+46))

        if st.blink_count > 0:
            dots = "●"*st.blink_count + "○"*max(0,3-st.blink_count)
            screen.blit(F_SM.render(f"Kirpma: {dots}", True, YELLOW), (CAM_X, ey+76))

    pos = (f"Satir {st.row+1}/{ROWS}   Sutun {st.col+1}/{rlen(st.row)}"
           f"   Harf: {st.letter}")
    screen.blit(F_SM.render(pos, True, TXT), (GL, GT+ROWS*(CH+GAP)+8))

    sy = SCREEN_H-34
    pygame.draw.rect(screen, SENT_BG, (0,sy-4,SCREEN_W,38))
    screen.blit(F_XS.render(st.status, True, TXT_D), (GL, sy+6))
    hint = ("Q=Cikis  C=Temizle  |  "
            "SagGozKirp=saga  SolGozKirp=sola  |  "
            "1xCift=sec  2x=asagi  3x=yukari")
    hs = F_XS.render(hint, True, TXT_D)
    screen.blit(hs, (SCREEN_W-hs.get_width()-8, sy+6))

    pygame.display.flip()

def loading(msg):
    screen.fill(BG)
    s = F_MD.render(msg, True, TXT)
    screen.blit(s, ((SCREEN_W-s.get_width())//2, SCREEN_H//2-20))
    pygame.display.flip(); pygame.event.pump()

def main():
    loading("Model kontrol ediliyor / indiriliyor...")
    download_model()
    loading("Kamera ve yapay zeka baslatiliyor...")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    if not cap.isOpened():
        loading("HATA: Kamera bulunamadi!"); time.sleep(3)
        pygame.quit(); sys.exit()

    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MPATH),
        running_mode=VisionTaskRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.55,
        min_face_presence_confidence=0.55,
        min_tracking_confidence=0.55,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    det = mp_vision.FaceLandmarker.create_from_options(opts)
    csf = None

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                cap.release(); det.close(); pygame.quit(); sys.exit()
            elif ev.type == pygame.KEYDOWN:
                if   ev.key == pygame.K_q: cap.release(); det.close(); pygame.quit(); sys.exit()
                elif ev.key == pygame.K_c: st.sentence=""; st.status="Temizlendi"

        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        H, W  = frame.shape[:2]
        now   = time.time()

        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        st.face_ok = bool(res.face_landmarks)

        if res.face_landmarks:
            lm = res.face_landmarks[0]

            l_raw = ear(lm, L_EAR, W, H)
            r_raw = ear(lm, R_EAR, W, H)
            st.smooth_ear(l_raw, r_raw)

            cur  = st.classify()
            prev = st.eye_state

            if prev != "OPEN" and cur == "OPEN":
                can = (now - st.last_move) > MOVE_COOLDOWN
                if prev == "R_WINK":
                    if can: st.right(); st.last_move = now
                elif prev == "L_WINK":
                    if can: st.left(); st.last_move = now
                elif prev == "BOTH":
                    st.blink_count += 1
                    st.blink_win = now

            st.eye_state = cur

            if (st.blink_win is not None
                    and cur == "OPEN"
                    and (now - st.blink_win) > BLINK_WIN):
                if st.blink_count > 0: st.on_blinks(st.blink_count)
                st.blink_count = 0; st.blink_win = None

            cv2.putText(frame,
                f"L:{st.l_s:.2f} R:{st.r_s:.2f} diff:{st.l_s-st.r_s:+.2f} [{cur}]",
                (6, H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200,220,255), 1)

            for iris in [[474,475,476,477],[469,470,471,472]]:
                cx = int(np.mean([lm[i].x*W for i in iris]))
                cy = int(np.mean([lm[i].y*H for i in iris]))
                cv2.circle(frame,(cx,cy),5,(22,95,230),-1)
                cv2.circle(frame,(cx,cy),9,(255,255,255),1)

        else:
            st.eye_state = "OPEN"
            st.blink_count = 0; st.blink_win = None
            st.status = "Yuz algilanamadi"

        csf = pygame.surfarray.make_surface(
            np.transpose(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),(1,0,2)))
        draw(csf)
        clock.tick(FPS)

if __name__ == "__main__":
    main()
